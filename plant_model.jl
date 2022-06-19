using DataFrames, Arrow, Dates, ShiftedArrays, Statistics, StatsBase
using JuMP, GLPK
using Suppressor
using LinearAlgebra
using Sockets, Serialization
using Logging

function simulate(
	;
    c::Dict{Symbol, Any},
    simulation_run_id::String,
	simulation_id::Int64,
	H2_energy_density::Float64,
	electrolyser_η::Float64,
    electrolyser_ramp_up_rate::Float64, # MW/h
    electrolyser_ramp_down_rate::Float64, # MW/h
    bop_load::Float64, # MW
	tank_wsc::Float64,
	# design_CF::Float64,
	electrolyser_capacity::Float64, # MW
	electrolyser_min_turndown::Float64, # MW
	power_portfolio::Vector{DataFrame},
	power_portfolio_scale::Vector{Float64},
	offtake_plan::Vector{Float64},
    planning_horizon_length::Int64,
	dest_path_and_filename_base::String,
	state_send_address::Union{String, Nothing} = nothing
)

    rate_factor_initial = 1
    rate_factor_step_down_size = 0.01

    # check that all power portfolio dfs have identical datetimes and store
    # those in a vector for convenience
    datetimes_by_power_portfolio = (x -> x.datetime).(power_portfolio)
    @assert (x -> reduce(&, map((y -> x[1] == y), x)))(
        datetimes_by_power_portfolio
    )
    datetimes = datetimes_by_power_portfolio[1]

    # check that all power portfolio dfs are of the same number of rows and
    # store that number as a atomic number
    T_by_power_portfolio = (x -> size(x)[1]).(power_portfolio)
    @assert (x -> reduce(&, map((y -> x[1] == y), x)))(
        T_by_power_portfolio
    )
    T = T_by_power_portfolio[1]

    # send configuration dictionary to dashboard
    if !isnothing(state_send_address) 
        @info join([
            "state_send_address = ";
            state_send_address;
            "; connecting\n";
        ])
        socket = connect(state_send_address, 8010)
        serialize(
            socket,
            (:c, c)
        )
    end
        
    power_portfolio = map(
        (y -> combine(y[1], :datetime, :price, :capacity_min => (z -> y[2] * z) => :capacity_min, :capacity_max => (z -> y[2] * z) => :capacity_max)),
        zip(
            power_portfolio,
            power_portfolio_scale
        )
    )


    @info join([
        "staring simulation_id = ";
        simulation_id;
        " with ";
        "T = ";
        T;
        "\n"
    ])
    
    history_electrolyser_power               = Array{Float32}(undef, T)
    history_production                       = Array{Float32}(undef, T)
    history_offtake                          = Array{Float32}(undef, T)
    history_tank_q                           = Array{Float32}(undef, T)
    history_electricity_price                = Array{Float32}(undef, (T, length(power_portfolio)))
    history_power_in                         = Array{Float32}(undef, (T, length(power_portfolio)))
    history_rate_factor                      = Array{Float32}(undef, T)
    history_model_termination_status         = Array{Int}(undef, T)
    history_1_day_average_electricity_price  = Array{Float32}(undef, T)
    history_10_day_average_electricity_price = Array{Float32}(undef, T)
    history_electrolyser_power_control_plan  = Array{Float32}(undef, (T, planning_horizon_length))

    @info "simulating"

    for t in 1:(T-planning_horizon_length)
        ti = Dates.value(Minute(Time(datetimes[t]) - Time(0, 0, 0))) ÷ 30 + 1
        duration = 0.5

        if ti == 1
            @info join([
                "simulation_id = ",
                simulation_id,
                " at start of ",
                Dates.Date(datetimes[t]),
                "\n"
            ])
        end

        if t == 1
            history_tank_q[1] = 0 # initial tank quantity (kg)
            previous_electrolyser_power = 0
        else
            # carry forward tank_q from end of previous period
            history_tank_q[t] = history_tank_q[t-1]
            previous_electrolyser_power = history_electrolyser_power[t-1]
        end

        power_portfolio_this_t = (x -> x[t, :]).(
                    power_portfolio,
        )
       
        history_electricity_price[t, :] = reduce(hcat, (x -> x[t, :price]).(power_portfolio))

        planning_horizon = t:(t+planning_horizon_length-1)

        if t > length(planning_horizon)
            n_rep_days = min(20, t ÷ 48)
            expected_price = reduce(
                hcat,
                (x -> reduce(vcat, (y -> lag(x, 48*y, default = x[1])[(end-length(planning_horizon)+1):end]).(1:n_rep_days)) ).(
                    (x -> x[1:(t-1), :price]).(power_portfolio)
                )
            )
            expected_capacity_min = reduce(
                hcat,
                (z -> mean(z, dims = 2)).((x -> reduce(hcat, (y -> lag(x, 48*y, default = x[1])[(end-length(planning_horizon)+1):end]).(1:n_rep_days)) ).((x -> x[1:(t-1), :capacity_min]).(power_portfolio)))
            )
            expected_capacity_max = reduce(
                hcat,
                (z -> mean(z, dims = 2)).((x -> reduce(hcat, (y -> lag(x, 48*y, default = x[1])[(end-length(planning_horizon)+1):end]).(1:n_rep_days)) ).((x -> x[1:(t-1), :capacity_max]).(power_portfolio)))
            )
        else
            n_rep_days = 1
            expected_price = reduce(
                hcat,
                fill(
                    (x -> x[t, :price]).(power_portfolio),
                    length(planning_horizon)
                )
            )'
            expected_capacity_min = reduce(
                hcat,
                fill(
                    (x -> x[t, :capacity_min]).(power_portfolio),
                    length(planning_horizon)
                )
            )'
            expected_capacity_max = reduce(
                hcat,
                fill(
                    (x -> x[t, :capacity_max]).(power_portfolio),
                    length(planning_horizon)
                )
            )'
        end

        rate_factor = rate_factor_initial

        global model = nothing
        global electrolyser_power_control_solution = nothing
        global power_in_solution = nothing
        global model_termination_status = nothing

        while isnothing(model) || (termination_status(model) != OPTIMAL)
            global model
            model = Model(GLPK.Optimizer)

            @variable(
                model,
                power_in[i = 1:length(planning_horizon), j = 1:length(power_portfolio)]
            ) # power in plan (a (length of planning horizon) × (size of power portfolio) matrix)

            @variable(
                model,
                electrolyser_power_control[i = 1:length(planning_horizon)],
                lower_bound = electrolyser_min_turndown,
                upper_bound = electrolyser_capacity
            ) # electrolyser power control plan (a (length of planning horizon) dimensional column vector) in MW
        
            @objective(
                model,
                Min,
                sum(reduce(vcat, fill(power_in, n_rep_days)) .* expected_price)
            ) # for multiple n_rep_days - minimise cost over all power portfolio items and over all representative days
        
            @constraint(
                model,
                c_power_in_lb[i = 1:length(planning_horizon), j = 1:length(power_portfolio)],
                reduce(vcat, fill(power_in, 1))[i, j] >= expected_capacity_min[i, j]
            ) # power in plan quantities must be greater than or equal to expected capacity for respective power portfolio item 
            
            @constraint(
                model,
                c_power_in_ub[i = 1:length(planning_horizon), j = 1:length(power_portfolio)],
                reduce(vcat, fill(power_in, 1))[i, j] <= expected_capacity_max[i, j]
            ) # power out plan quantities must be greater than or equal to expected capacity for respective power portfolio item 
            
            @constraint(
                model,
                c_power_balance,
                ([electrolyser_power_control fill(bop_load, length(planning_horizon)) -power_in] * ones(length(power_portfolio)+2)) .== 0
            ) # instantaneous power balance
            
            @constraint(
                model,
                c_storage_lb,
                ( history_tank_q[t] * ones(length(planning_horizon))
                    + cumsum(electrolyser_power_control) * electrolyser_η * duration * (60 * 60) / (H2_energy_density)
                    - cumsum(rate_factor*offtake_plan[t:t+(length(planning_horizon)-1)])
                ) .>= 0
            ) # storage must be greater than or equal to 0
            
            @constraint(
                model,
                c_storage_ub,
                ( history_tank_q[t] * ones(length(planning_horizon))
                    + cumsum(electrolyser_power_control) * electrolyser_η * duration * (60 * 60) / (H2_energy_density)
                    - cumsum(rate_factor*offtake_plan[t:t+(length(planning_horizon)-1)])
                ) .<= tank_wsc
            ) # storage must be less than or equal to `tank_wsc`

            @constraint(
                model,
                c_ramp_up_first_element,
                electrolyser_power_control[1] - previous_electrolyser_power <= electrolyser_ramp_up_rate
            ) #  first element in electrolyser power control plan must be no more than ramp up rate greater than previous (actual) electrolyser power
            
            @constraint(
                model,
                c_ramp_up_subsequent_elements[i = 2:length(planning_horizon)],
                electrolyser_power_control[i] - electrolyser_power_control[max(i-1, 1)] <= electrolyser_ramp_up_rate # MW / trading interval
            ) # consecutive elements in electrolyser power control plan must increase by no more than ramp up rate
            
            @constraint(
                model,
                c_ramp_down_first_element,
                electrolyser_power_control[1] - previous_electrolyser_power >= -electrolyser_ramp_down_rate # MW / trading interval
            ) #  first element in electrolyser power control plan must be no more than ramp up rate greater than previous (actual) electrolyser power
            
            @constraint(
                model,
                c_ramp_down_subsequent_elements[i = 2:length(planning_horizon)],
                electrolyser_power_control[i] - electrolyser_power_control[max(i-1, 1)] >= -electrolyser_ramp_down_rate # MW / trading interval
            ) # consecutive elements in electrolyser power control plan must decrease by no more than ramp down rate
            
            @suppress optimize!(model)

            if (termination_status(model) != OPTIMAL)
                if rate_factor > 0
                    rate_factor_new = max(0, rate_factor - rate_factor_step_down_size)
                    @warn join(["reducing rate factor from ", rate_factor, " to ", rate_factor_new])
                    rate_factor = rate_factor_new
                else
                    throw(ErrorException("No acceptable rate factor gives an optimal solution."))
                end
            end
            global electrolyser_power_control_solution = value.(electrolyser_power_control)
            global power_in_solution = value.(power_in)
            global model_termination_status = Int(termination_status(model))
        end

        history_offtake[t] = min(
            history_tank_q[t],
            offtake_plan[t]
        )
        @assert history_tank_q[t] - history_offtake[t] >= 0
        history_tank_q[t] -= history_offtake[t]

        history_electrolyser_power[t] = electrolyser_power_control_solution[1]
        history_electrolyser_power_control_plan[t, :] = electrolyser_power_control_solution 
        
        history_power_in[t, :] = power_in_solution[1, :]

        history_production[t] = min(
            (electrolyser_power_control_solution[1] * electrolyser_η * duration) * (60 * 60) / (H2_energy_density), # (MWh) * (J/Wh) / (MJ/kg) = kg
            tank_wsc - history_tank_q[t]
        )

        history_rate_factor[t] = rate_factor
        history_model_termination_status[t] = model_termination_status 
        
        history_1_day_average_electricity_price[t] = sum(history_power_in[max(t-48, 1):t, :] .* history_electricity_price[max(t-48, 1):t, :]) / sum(history_power_in[max(t-48, 1):t, :])
        history_10_day_average_electricity_price[t] = sum(history_power_in[max(t-480, 1):t, :] .* history_electricity_price[max(t-480, 1):t, :]) / sum(history_power_in[max(t-480, 1):t, :])

        history_tank_q[t] += history_production[t]

        if ti == 48 
            @info join([
                "simulation_id = ",
                simulation_id,
                " at end of ",
                Dates.Date(datetimes[t]),
                "\n"
            ])
            @info join([
                "sum of last 48 history_production: ",
                sum(history_production[(t-48+1):t]),
                "\n"
            ])
            @info join([
                "sum of last 48 history_offtake: ",
                sum(history_offtake[(t-48+1):t]),
                "\n"
            ])
            @info join([
                "sum of last 48 offtake_plan: ",
                sum(offtake_plan[(t-48+1):t]),
                "\n"
            ])
        end

        if !isnothing(state_send_address)
            serialize(
                socket,
                (
                    :model_state,
                    (
                        t,
                        datetimes[max(1, (t-96+1)):t],
                        history_electricity_price[max(1, (t-96+1)):t, :],
                        string(termination_status(model)),
                        expected_price,
                        history_power_in,
                        history_1_day_average_electricity_price[max(1, (t-96+1)):t],
                        history_10_day_average_electricity_price[max(1, (t-96+1)):t],
                        history_tank_q[max(1, (t-96+1)):t],
                        history_electrolyser_power[max(1, (t-96+1)):t],
                        expected_price,
                        n_rep_days,
                        collect(electrolyser_power_control_solution),
                        history_power_in[max(1, (t-96+1)):t, :],
                        expected_capacity_min,
                        expected_capacity_max,
                        electrolyser_capacity,
                        tank_wsc,
                        Dict(
                            :simulation_id => simulation_id,
                            :rate_factor => rate_factor,
                            :model_termination_status => model_termination_status
                        )
                    )
                )
            )
        end
    end
    
    io_buffer = IOBuffer()
    Arrow.write(
        io_buffer,
        stack(
            reduce(
                hcat,
                [
                    DataFrame(datetime = datetimes),
                    DataFrame(history_electrolyser_power_control_plan, :auto)
                ]
            ),
            variable_name = :lead_tis,
            value_name = :electrolyser_power
        )
    )
    (if !isnothing(match(r"^s3://.+$", dest_path_and_filename_base))
        robust_write_s3
    else
        write
    end)(
        join([dest_path_and_filename_base, "_production_plan.feather"]),
        io_buffer
    ) 

    io_buffer = IOBuffer()
    Arrow.write(
        io_buffer,
        innerjoin(
            stack(
                reduce(
                    hcat,
                    [
                        DataFrame(datetime = datetimes),
                        DataFrame(history_power_in, :auto)
                    ]
                ),
                variable_name = :power_portfolio_item,
                value_name = :quantity
            ),
            stack(
                reduce(
                    hcat,
                    [
                        DataFrame(datetime = datetimes),
                        DataFrame(history_electricity_price, :auto)
                    ]
                ),
                variable_name = :power_portfolio_item,
                value_name = :price
            ),
            on = [:datetime, :power_portfolio_item]
        )
    )
    (if !isnothing(match(r"^s3://.+$", dest_path_and_filename_base))
        robust_write_s3
    else
        write
    end)(
        join([dest_path_and_filename_base, "_power.feather"]),
        io_buffer
    ) 

    io_buffer = IOBuffer()
    Arrow.write(
        io_buffer,
        DataFrame(
            datetime = datetimes,
            tank_q = history_tank_q,
            production = history_production,
            rate_factor = history_rate_factor,
            model_termination_status = history_model_termination_status
        ) 
    )
    (if !isnothing(match(r"^s3://.+$", dest_path_and_filename_base))
        robust_write_s3
    else
        write
    end)(
        join([dest_path_and_filename_base, "_production.feather"]),
        io_buffer
    ) 
end