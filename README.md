## Experiment on Images

Train the framework for generation using `mnist_wgan_inv.py`

or use pre-trained framework located in `./models`

Then generate natural adversaries using `mnist_natural_adversary.py`

Classifiers:
- Random Forest (90.45%), `--classifier rf`
- LeNet (98.71%), `--classifier lenet`

Algorithms:
- iterative stochastic search, `--iterative`
- hybrid shrinking search (default)

Output samples are located in `./examples`

### Structure
mnist_wgan_inv.py trains a GAN and saves the generator and inverter into models.
It saves checkpoints into training_checkpoints_mnist and saves task progress over
epochs in images.

lenet_mnist creates a mnist classifier and saves it into models.

mnist_natural_adversary takes the classifier, generator, and inverter, and finds
adversarial images which it saves into examples.

search has the algorithms for finding adversarial examples.

#### Acknowledgment
`./tflib` is based on [Gulrajani et al., 2017](https://github.com/igul222/improved_wgan_training)
