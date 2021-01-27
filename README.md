# vis1d
A one-dimensional visual environment for modeling complex visual phenomena without slow rendering times.

Running env/env1d.py renders a simple example environment episode.

An LSTM model and a training loop with single-step predictive learning have been added as well. The model currently learns to very roughly track a single flat object, but much of the basic infrastructure is in place to play around.

Main dependencies:
* Python 3 (tested on 3.8)
* torch >= 1.4
