import statistics
import numpy as np

def tabu_search(func, x0, tenure=2, max_iter=100, bounds=None, neighbors_size=10):
    num_dimensions = len(x0)
    current_solution = list(x0)  # Use list for mutability
    best_solution = list(current_solution)
    best_objective_value = func(current_solution)
    tabu_list = {}
    all_objective_values = []

    for _ in range(max_iter):
        neighbors = []

        # Generate neighbors
        for _ in range(neighbors_size):
            new_solution_candidate = list(current_solution)
            
            # Select a random dimension to perturb
            dimension_index = np.random.randint(0, num_dimensions)
            perturbation_delta = np.random.uniform(-0.5, 0.5)
            new_solution_candidate[dimension_index] += perturbation_delta

            # Apply bounds if provided
            if bounds:
                new_solution_candidate[dimension_index] = max(bounds[0], min(bounds[1], new_solution_candidate[dimension_index]))

            # Evaluate the new solution
            new_objective_value = func(new_solution_candidate)
            all_objective_values.append(new_objective_value)
            
            # Store move details, new solution, and its objective value
            neighbors.append(((dimension_index, perturbation_delta), new_solution_candidate, new_objective_value))
        
        # Sort neighbors by their objective value (ascending for minimization)
        neighbors.sort(key=lambda t: t[2])

        best_neighbor_move = None
        best_neighbor_solution = None
        best_neighbor_objective_value = None

        # Find the best non-tabu neighbor or an aspiration criteria satisfying move
        for move_details, neighbor_solution, neighbor_objective_value in neighbors:
            # A move is defined by (dimension_index, perturbation_delta)
            # The reverse move is (dimension_index, -perturbation_delta) which would undo the perturbation
            reverse_move_identifier = (move_details[0], -move_details[1]) 
            
            # Aspiration Criteria: if the neighbor is better than the global best, accept it even if tabu
            if reverse_move_identifier not in tabu_list or neighbor_objective_value < best_objective_value:
                best_neighbor_move = move_details
                best_neighbor_solution = neighbor_solution
                best_neighbor_objective_value = neighbor_objective_value
                break # Found an acceptable move, take it

        # If no acceptable move is found, stop the search
        if best_neighbor_move is None:
            break

        # Calculate the reverse move identifier from the chosen move
        reverse_move_identifier = (best_neighbor_move[0], -best_neighbor_move[1])

        # Update current solution
        current_solution = list(best_neighbor_solution)
        current_objective_value = best_neighbor_objective_value

        # Update global best if a better solution is found
        if current_objective_value < best_objective_value:
            best_solution = list(current_solution)
            best_objective_value = current_objective_value

        # Decrement tenure for all moves in the tabu list
        tabu_list = {move: t - 1 for move, t in tabu_list.items() if t - 1 > 0}
        
        # Add the reverse of the chosen move to the tabu list
        # This prevents immediately reversing the last move
        tabu_list[reverse_move_identifier] = tenure
    
    # Calculate statistics of all explored objective values
    avg_f = sum(all_objective_values) / len(all_objective_values) if all_objective_values else 0
    median_f = statistics.median(all_objective_values) if all_objective_values else 0
    max_f = max(all_objective_values) if all_objective_values else 0
    
    return best_solution, best_objective_value, avg_f, median_f, max_f



