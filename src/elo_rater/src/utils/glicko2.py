import numpy as np

def g(RDi):
    q = np.log(10) / 400
    
    term  = (3 * q**2 * RDi**2) / np.pi**2
    
    result = 1 / np.sqrt( 1 + term)
    return result

def E(RDi, r0, ri):
    exponent = (g(RDi) * (r0 - ri)) / -400
    e_out = 1 / (1 + 10**exponent)
    return e_out

def d_square(RDs, my_rank, opponent_ranks):
    q = np.log(10) / 400
    total_sum = 0
    for i in range(len(opponent_ranks)):
        RDi = RDs[i]
        opponent_rank = opponent_ranks[i]
        
        value = g(RDi)**2 * \
                E(RDi, my_rank, opponent_rank) * \
                (1 - E(RDi, my_rank, opponent_rank))
        
        total_sum += value
    denominator = q**2 * value
    
    return 1 / denominator


def rank(my_rank, 
         my_old_rank_deviation,
         missed_rank_steps,
         games,
         opponent_ranks,
         opponent_rank_deviations):
    c = 34.6
    my_rank_deviation = np.sqrt(my_old_rank_deviation**2 + c**2 *missed_rank_steps)
    q = np.log(10) / 400
    d_square_value = d_square(opponent_rank_deviations, 
                              my_rank, 
                              opponent_ranks)

    first_factor = q / ((1/my_rank_deviation**2) + (1/d_square_value)) 

    total_sum = 0
    for i in range(len(games)):
        game_result = games[i]
        RDi = opponent_rank_deviations[i]
        opponent_rank = opponent_ranks[i]

        value = g(RDi) * (game_result - E(RDi, my_rank, opponent_rank))
        total_sum += value

    new_rank = my_rank + first_factor*total_sum
    a = 1/(my_rank_deviation**2)
    b = 1/d_square_value
    new_rank_deviation = np.sqrt((a+b)**-1)
    return round(new_rank), round(new_rank_deviation)
