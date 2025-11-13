from typing import List, Sequence, Callable, Tuple,Dict,Any

#ann packages
from src.ann.builder import build_network
from src.ann.loss import mean_absolute_error
from src.ann.network import Network

#pso packages
from src.pso import PSO, PSOConfig
from src.ann import ANNConfig



"""Build ANN architecture that PSO will optimize.
   For course work
          * 8 inputs
          * 1 output
          * 2 hidden layers
             -units 16 activation-relu
             -units 8 activation -tanh """


"""
    Wrap the ANN as a PSO objective.
    PSO will call this with a flat vector of params, we:
      1) set ANN params
      2) run ANN on all data
      3) compute loss
    and return that loss.
    """
#region make ann objective
def make_ann_objective(
    network: Network,                                                                #predefined ANN network
    x: Sequence[Sequence[float]],                                                    #input_values
    y: Sequence[float],                                                              #output_values
    loss_fn: Callable[[List[float], List[float]], float] = mean_absolute_error,      #loss function
) -> Callable[[List[float]], float]:


    def objective(position: List[float]) -> float:
        # 1) put particle's weight and bias into the ANN
        network.set_parameters(position)

        # 2) forward-pass all samples with current ANN
        predictions: List[float] = []
        for x_input in x:
            out = network.forward_algorithm(x_input)
            predictions.append(out[0])        # take the single output

        # 3) lower loss = better particle
        return loss_fn(list(y), predictions)

    return objective

#endregion


""" train ANN with PSO 
      - build the network
    - create objective for PSO
    - run PSO
    - return (trained_network, best_loss)"""

def train_ann_with_pso(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
    pso_config: PSOConfig = PSOConfig(),
    ann_config: ANNConfig = ANNConfig(),


) -> Tuple[Network, float,List[float]]:

    # 1) fix ANN architecture for this PSO run

    net = build_network(ann_config.input_dim,ann_config.hidden_layers,ann_config.output_dim,ann_config.output_activation)

    # 2) PSO dimension = how many scalars ANN needs
    dim = net.number_parameters()

    # 3) build objective function that PSO can call
    objective = make_ann_objective(net, x_train, y_train)

    # 4) PSO config
    cfg = PSOConfig(
        swarm_size=pso_config.swarm_size,
        iterations=pso_config.iterations,
        bounds=pso_config.bounds,  # every weight/bias must stay in this range
        minimize=pso_config.minimize,
        k_informants=pso_config.k_informants,
    )

    # 5) create PSO instance with our objective
    pso = PSO(
        dimension=dim,
        fitness_fn=objective,
        config=cfg,
    )

    # 6) run optimisation
    best_position, best_fitness,history = pso.run(verbose=True)

    # 7) load the best weights into the ANN so caller can use it
    net.set_parameters(best_position)

    return net, best_fitness,history
