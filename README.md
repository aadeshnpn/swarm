# Swarm

A simple swarm simulator to simulate collective behaviors using grammatical evolution and Behavior Trees. Users can use the exisiting primitive behaviors or can define new behaviors with ease and see collective behaviors either manually building behavior tree or let grammatical eovlution do the trick to evolve collective behaviors. This project is based on mesa framework. Brief feature list:
* Notable primitive behaviors already coded
* Easy to use and visualize
* Uses behaviors trees
* Grammatical evolution for evolving collective behaviors 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
pip install matplotlib numpy scipy scikit-learn pandas
```

### Installing

This swarm framework depends on PonyGE2 for grammatical evolution and py_trees for Behavior trees. You need to clone these framework from below location

PonyGE2

```
git clone https://github.com/aadeshnpn/PonyGE2
cd PonyGE2
pip install .
```

And for py_trees

```
git clone https://github.com/aadeshnpn/py_trees
cd py_trees
pip install .
```

Now all the dependencies are installed

clone the swarm repo 
```
git clone https://github.com/aadeshnpn/swarm
cd swarm
pip install .
```
Now you have the swarm framework installed. 

## Running the tests

All the tests files are located under test folder. Just traverse to the folder and run
```
nosetest *.py
```

## Contributing

Just submit the pull requests to us.

## Authors

* **Aadesh Neupane** - *Initial work* - [Aadeshnpn](https://github.com/aadeshnpn)

See also the list of [contributors](https://github.com/aadeshnpn/swarm/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Mesa team

