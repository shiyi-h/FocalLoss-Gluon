from mxnet.gluon import loss
import mxnet as mx
from mxnet.util import is_np_array
from mxnet.base import numeric_types
import numpy as np


def _apply_gamma_alpha(F, pt, gamma=None, alpha=None):
    if gamma is not None:
        assert isinstance(gamma, numeric_types), "gamma must be a number"
        if is_np_array():
            loss = -F.np.power(1-pt, gamma)*F.np.log(pt)
        else:
            loss = -F.power(1-pt, gamma)*F.log(pt)
    else:
        loss = -pt.log()
    if alpha is not None:
        if isinstance(alpha, numeric_types):
            loss = loss*alpha
        else:
            if is_np_array():
                loss = loss*alpha
            else:
                loss = F.broadcast_mul(loss, alpha)
    return loss

def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    if F is mx.ndarray:
        return x.reshape(y.shape)
    elif mx.util.is_np_array():
        F = F.npx
    return F.reshape_like(x, y)


class FocalLoss(loss.Loss):
    def __init__(self,gamma=2, alpha=0.25, axis=-1, sparse_label=True,
                 from_logits=False, weight=None, batch_axis=0, **kwargs):
        super(FocalLoss,self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self.epsilon = 1e-10
        self._gamma = gamma
        self._alpha = alpha
        assert isinstance(self._alpha,(numeric_types,list,mx.nd.NDArray,np.ndarray))
        if isinstance(self._alpha, list):
            if is_np_array():
                self._alpha = np.array(self._alpha)
            else:
                self._alpha = mx.nd.array(self._alpha)

    def __call__(self, *args):
       alpha = self._alpha
       res = super(FocalLoss, self).__call__(*args, alpha)
       return res

    def hybrid_forward(self, F, pt, label, _alpha):
        if is_np_array():
            softmax = F.npx.softmax
            pick = F.npx.pick
            ones_like = F.np.ones_like
        else:
            softmax = F.softmax
            pick = F.pick
            ones_like = F.ones_like
        if not self._from_logits:
            pt = softmax(pt, axis=self._axis)
        if not isinstance(_alpha, mx.base.numeric_types):
            _alpha = F.broadcast_mul(ones_like(pt),_alpha)
            _alpha = pick(_alpha, label, axis=self._axis, keepdims=True)

        if self._sparse_label:
            pt = pick(pt, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pt)
            pt = (pt * label).sum(axis=self._axis, keepdims=True)

        loss = _apply_gamma_alpha(F, pt, self._gamma, _alpha)

        if is_np_array():
            if F is mx.ndarray:
                return loss.mean(axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return loss.mean(axis=self._batch_axis, exclude=True)

if __name__ == '__main__':
    x = mx.nd.array([[1,2,3],[4,5,6],[9,8,7]])
    label = mx.nd.array([[1],[2],[0]])
    alpha = [1,2,3]
    loss = FocalLoss(alpha=alpha)
    loss.hybridize()
    print(loss(x,label))
