immutable RankInfo
    label::String
    count::Int
    score_values::Array{Int}
    reward::Int
    penalty::Int
end

type ScoreTransitionMatrix
    transition_matrix::Array{Float64, 2}
    rank_info::RankInfo
end

type ScoreSteadyState
    probabilities::Array{Float64}
    rank_info::RankInfo
end

function steady_state(transition_matrix::Array{Float64, 2})
    values, vectors = eig(transition_matrix')
    for i=1:length(values)
        if abs(values[i] - 1.0) < 1e-9
            vec = real(vectors[:,i])
            return vec/sum(vec)
        end
    end
    return zeros(length(values))
end

function score_steady_state(score_transition_matrix::ScoreTransitionMatrix)
    steady_state_vector = steady_state(score_transition_matrix.transition_matrix)
    return ScoreSteadyState(steady_state_vector, score_transition_matrix.rank_info)
end

function rank_matrix_from_steady_states(steady_states::Array{ScoreSteadyState})
    count = sum([steady_states[i].rank_info.count for i=1:length(steady_states)])

    rank_matrix = eye(count)

    i = 1
    for idx=1:length(steady_states)
        rank_info = steady_states[idx].rank_info
        probabilities = steady_states[idx].probabilities
        for offset=1:rank_info.count
            if i > 1
                for j=1:length(rank_info.score_values)
                    if rank_info.score_values[j] + rank_info.penalty < rank_info.score_values[1]
                        rank_matrix[i,i-1] += 0.5 * probabilities[j]
                    end
                end
                rank_matrix[i,i] -= rank_matrix[i,i-1]
            end
            if i < count
                for j=1:length(rank_info.score_values)
                    if rank_info.score_values[j] + rank_info.reward >= 100
                        rank_matrix[i,i+1] += 0.5 * probabilities[j]
                    end
                end
                rank_matrix[i,i] -= rank_matrix[i,i+1]
            end
            i = i + 1
        end
    end

    return rank_matrix
end

function report_score_steady_states(score_steady_states::Array{ScoreSteadyState})
    for i=1:length(score_steady_states)
        ss = score_steady_states[i]
        println(ss.rank_info.label, ": ", ss.probabilities)
    end
end

function report_rank_steady_state(rank_steady_state)
    println("Rank steady state: ", rank_steady_state)

    println("C: ", round(100 * sum(rank_steady_state[1:3]), 2), "%")
    println("B: ", round(100 * sum(rank_steady_state[4:6]), 2), "%")
    println("A: ", round(100 * sum(rank_steady_state[7:9]), 2), "%")
end

function default_score_transitions()
                            # 0  10  20  30  40  50  60  70  80  90
    c_minus_matrix = Float64[0.5  0  0.5  0   0   0   0   0   0   0; #  0
                             0.5  0   0  0.5  0   0   0   0   0   0; # 10
                              0  0.5  0   0  0.5  0   0   0   0   0; # 20
                              0   0  0.5  0   0  0.5  0   0   0   0; # 30
                              0   0   0  0.5  0   0  0.5  0   0   0; # 40
                              0   0   0   0  0.5  0   0  0.5  0   0; # 50
                              0   0   0   0   0  0.5  0   0  0.5  0; # 60
                              0   0   0   0   0   0  0.5  0   0  0.5;# 70
                              0   0   0   0   0   0   0  1.0  0   0; # 80
                              0   0   0   0   0   0   0  0.5 0.5  0 ]# 90

    c_minus = ScoreTransitionMatrix(c_minus_matrix, RankInfo("C-", 1, [10*i for i=0:9], 20, -10))

                      # 0   5  10   15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90  95
    c_matrix = Float64[ 0   0   0   0.5  0   0  0.5  0   0   0   0   0   0   0   0   0   0   0   0   0; #  0
                        0   0   0    0  0.5  0  0.5  0   0   0   0   0   0   0   0   0   0   0   0   0; #  5
                       0.5  0   0    0   0  0.5  0   0   0   0   0   0   0   0   0   0   0   0   0   0; # 10
                        0  0.5  0    0   0   0  0.5  0   0   0   0   0   0   0   0   0   0   0   0   0; # 15
                        0   0  0.5   0   0   0   0  0.5  0   0   0   0   0   0   0   0   0   0   0   0; # 20
                        0   0   0   0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0   0   0   0; # 25
                        0   0   0    0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0   0   0; # 30
                        0   0   0    0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0   0; # 35
                        0   0   0    0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0; # 40
                        0   0   0    0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0; # 45
                        0   0   0    0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0; # 50
                        0   0   0    0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0; # 55
                        0   0   0    0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0; # 60
                        0   0   0    0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0; # 65
                        0   0   0    0   0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0; # 70
                        0   0   0    0   0   0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0; # 75
                        0   0   0    0   0   0   0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5; # 80
                        0   0   0    0   0   0   0   0   0   0   0   0   0   0  0.5 0.5  0   0   0   0; # 85
                        0   0   0    0   0   0   0   0   0   0   0   0   0   0  0.5  0  0.5  0   0   0; # 90
                        0   0   0    0   0   0   0   0   0   0   0   0   0   0  0.5  0   0  0.5  0   0] # 95

    c = ScoreTransitionMatrix(c_matrix, RankInfo("C", 1, [5*i for i=0:19], 15, -10))

    c_plus_matrix = zeros(50, 50)

    for i=1:50
        if i+6 <= 50
            c_plus_matrix[i,i+6] = 0.5 # +12
        else
            c_plus_matrix[i,36] = 0.5 # 70
        end

        if i-5 >= 1
            c_plus_matrix[i,i-5] = 0.5 # -10
        else
            c_plus_matrix[i,16] = 0.5 # 30
        end
    end

    c_plus = ScoreTransitionMatrix(c_plus_matrix, RankInfo("C+/B-", 2, [2*i for i=0:49], 12, -10))

                      # 0  10  20  30  40  50  60  70  80  90
    b_matrix = Float64[ 0  0.5  0  0.5  0   0   0   0   0   0; #  0
                       0.5  0  0.5  0   0   0   0   0   0   0; # 10
                        0  0.5  0  0.5  0   0   0   0   0   0; # 20
                        0   0  0.5  0  0.5  0   0   0   0   0; # 30
                        0   0   0  0.5  0  0.5  0   0   0   0; # 40
                        0   0   0   0  0.5  0  0.5  0   0   0; # 50
                        0   0   0   0   0  0.5  0  0.5  0   0; # 60
                        0   0   0   0   0   0  0.5  0  0.5  0; # 70
                        0   0   0   0   0   0   0  0.5  0  0.5;# 80
                        0   0   0   0   0   0   0  0.5 0.5  0];# 90

    b = ScoreTransitionMatrix(b_matrix, RankInfo("B/B+/A-/A", 4, [10*i for i=0:9], 10, -10))

                           # 0  10  20  30  40  50  60  70  80  90 100
    a_plus_matrix = Float64[ 0  0.5  0  0.5  0   0   0   0   0   0   0; #  0
                            0.5  0  0.5  0   0   0   0   0   0   0   0; # 10
                             0  0.5  0  0.5  0   0   0   0   0   0   0; # 20
                             0   0  0.5  0  0.5  0   0   0   0   0   0; # 30
                             0   0   0  0.5  0  0.5  0   0   0   0   0; # 40
                             0   0   0   0  0.5  0  0.5  0   0   0   0; # 50
                             0   0   0   0   0  0.5  0  0.5  0   0   0; # 60
                             0   0   0   0   0   0  0.5  0  0.5  0   0; # 70
                             0   0   0   0   0   0   0  0.5  0  0.5  0;# 80
                             0   0   0   0   0   0   0   0  0.5  0  0.5;# 90
                             0   0   0   0   0   0   0   0   0  0.5 0.5];# 100

    a_plus = ScoreTransitionMatrix(a_plus_matrix, RankInfo("A+", 1, [10*i for i in 0:10], 10, -10))

    return ScoreTransitionMatrix[ c_minus, c, c_plus, b, a_plus ]
end

function proposed_score_transitions()
                            # 0  10  20  30  40  50  60  70  80  90
    c_minus_matrix = Float64[0.5  0  0.5  0   0   0   0   0   0   0; #  0
                             0.5  0   0  0.5  0   0   0   0   0   0; # 10
                              0  0.5  0   0  0.5  0   0   0   0   0; # 20
                              0   0  0.5  0   0  0.5  0   0   0   0; # 30
                              0   0   0  0.5  0   0  0.5  0   0   0; # 40
                              0   0   0   0  0.5  0   0  0.5  0   0; # 50
                              0   0   0   0   0  0.5  0   0  0.5  0; # 60
                              0   0   0   0   0   0  0.5  0   0  0.5;# 70
                              0   0   0   0   0   0   0  1.0  0   0; # 80
                              0   0   0   0   0   0   0  0.5 0.5  0 ]# 90

    c_minus = ScoreTransitionMatrix(c_minus_matrix, RankInfo("C-", 1, [10*i for i=0:9], 20, -10))

                      #  5  10   15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90  95
    c_matrix0= Float64[  0   0    0  0.5  0  0.5  0   0   0   0   0   0   0   0   0   0   0   0   0; #  5
                         0   0    0   0  0.5 0.5  0   0   0   0   0   0   0   0   0   0   0   0   0; # 10
                        0.5  0    0   0   0  0.5  0   0   0   0   0   0   0   0   0   0   0   0   0; # 15
                         0  0.5   0   0   0   0  0.5  0   0   0   0   0   0   0   0   0   0   0   0; # 20
                         0   0   0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0   0   0   0; # 25
                         0   0    0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0   0   0; # 30
                         0   0    0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0   0; # 35
                         0   0    0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0   0; # 40
                         0   0    0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0   0; # 45
                         0   0    0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0   0; # 50
                         0   0    0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0   0; # 55
                         0   0    0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0   0; # 60
                         0   0    0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0   0; # 65
                         0   0    0   0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0   0; # 70
                         0   0    0   0   0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5  0; # 75
                         0   0    0   0   0   0   0   0   0   0   0   0   0  0.5  0   0   0   0  0.5; # 80
                         0   0    0   0   0   0   0   0   0   0   0   0   0  0.5 0.5  0   0   0   0; # 85
                         0   0    0   0   0   0   0   0   0   0   0   0   0  0.5  0  0.5  0   0   0; # 90
                         0   0    0   0   0   0   0   0   0   0   0   0   0  0.5  0   0  0.5  0   0] # 95

    c = ScoreTransitionMatrix(c_matrix0, RankInfo("C", 1, [5*i for i=1:19], 15, -10))

    c_plus_matrix0 = zeros(49, 49)

    for i=1:49
        if i+6 <= 49
            c_plus_matrix0[i,i+6] = 0.5 # +12
        else
            c_plus_matrix0[i,35] = 0.5 # 70
        end

        if i-5 >= 1
            c_plus_matrix0[i,i-5] = 0.5 # -10
        else
            c_plus_matrix0[i,15] = 0.5 # 30
        end
    end

    c_plus = ScoreTransitionMatrix(c_plus_matrix0, RankInfo("C+/B-", 2, [2*i for i=1:49], 12, -10))

                      # 10  20  30  40  50  60  70  80  90
    b_matrix0= Float64[  0  0.5 0.5  0   0   0   0   0   0; # 10
                        0.5  0  0.5  0   0   0   0   0   0; # 20
                         0  0.5  0  0.5  0   0   0   0   0; # 30
                         0   0  0.5  0  0.5  0   0   0   0; # 40
                         0   0   0  0.5  0  0.5  0   0   0; # 50
                         0   0   0   0  0.5  0  0.5  0   0; # 60
                         0   0   0   0   0  0.5  0  0.5  0; # 70
                         0   0   0   0   0   0  0.5  0  0.5;# 80
                         0   0   0   0   0   0  0.5 0.5  0];# 90

    b = ScoreTransitionMatrix(b_matrix0, RankInfo("B/B+/A-/A", 4, [10*i for i=1:9], 10, -10))

                           # 10  20  30  40  50  60  70  80  90 100
    a_plus_matrix0= Float64[  0  0.5 0.5  0   0   0   0   0   0   0; # 10
                             0.5  0  0.5  0   0   0   0   0   0   0; # 20
                              0  0.5  0  0.5  0   0   0   0   0   0; # 30
                              0   0  0.5  0  0.5  0   0   0   0   0; # 40
                              0   0   0  0.5  0  0.5  0   0   0   0; # 50
                              0   0   0   0  0.5  0  0.5  0   0   0; # 60
                              0   0   0   0   0  0.5  0  0.5  0   0; # 70
                              0   0   0   0   0   0  0.5  0  0.5  0;# 80
                              0   0   0   0   0   0   0  0.5  0  0.5;# 90
                              0   0   0   0   0   0   0   0  0.5 0.5];# 100

    a_plus = ScoreTransitionMatrix(a_plus_matrix0, RankInfo("A+", 1, [10*i for i in 1:10], 10, -10))

    return ScoreTransitionMatrix[ c_minus, c, c_plus, b, a_plus ]
end

function solve_and_report(score_transitions::Array{ScoreTransitionMatrix})
    score_steady_states = [score_steady_state(score_transitions[i]) for i in 1:length(score_transitions)]
    report_score_steady_states(score_steady_states)

    rank_matrix = rank_matrix_from_steady_states(score_steady_states)

    println("Rank transition matrix: ", rank_matrix)

    rank_steady_state = steady_state(rank_matrix)

    report_rank_steady_state(rank_steady_state)
end

println("\n *** Current Splatoon algorithm (demote when score < 0) *** ")
solve_and_report(default_score_transitions())

println("\n *** Proposed fix (demote when score = 0) *** ")
solve_and_report(proposed_score_transitions())
