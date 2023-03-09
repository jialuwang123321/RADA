from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional

import numpy as np

from AT.config import ART_NUMPY_DTYPE
from AT.attack import EvasionAttack
from AT.estimator import BaseEstimator, LossGradientsMixin
from AT.classifier import (
    ClassifierGradients,
    ClassifierMixin,
)
from AT.utils import (
    compute_success,
    get_labels_np_array,
    random_sphere,
    projection,
    check_and_transform_label_format,
)

logger = logging.getLogger(__name__)


class RADA_Attack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: ClassifierGradients,
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        minimal: bool = False,
        optuna_pow:float=1.5,
        perturb_threshold:int=1,
    ) -> None:
        """
        Create a :class:`.FastGradientMethod` instance.

        :param estimator: A trained classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :param eps: Attack step size (input variation).
        :param eps_step: Step size of input variation for minimal perturbation computation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param num_random_init: Number of random initialisations within the epsilon ball. For random_init=0 starting at
            the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param minimal: Indicates if computing the minimal perturbation (True). If True, also define `eps_step` for
                        the step size and eps for the maximum perturbation.
        """
        super(RADA_Attack, self).__init__(estimator=estimator)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.minimal = minimal
        self._project = True
        self.optuna_pow = optuna_pow
        self.perturb_threshold = perturb_threshold
        RADA_Attack._check_params(self)

    def _minimal_perturbation(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Iteratively compute the minimal perturbation necessary to make the class prediction change. Stop when the
        first adversarial example was found.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :return: An array holding the adversarial examples.
        """
        adv_x = x.copy()

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(adv_x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = (
                batch_id * self.batch_size,
                (batch_id + 1) * self.batch_size,
            )
            batch = adv_x[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels, mask_batch)

            # Get current predictions
            active_indices = np.arange(len(batch))
            current_eps = self.eps_step
            while active_indices.size > 0 and current_eps <= self.eps:
                # Adversarial crafting
                current_x = self._apply_perturbation(x[batch_index_1:batch_index_2], perturbation, current_eps)
                # Update
                batch[active_indices] = current_x[active_indices]
                adv_preds = self.estimator.predict(batch)
                # If targeted active check to see whether we have hit the target, otherwise head to anything but
                if self.targeted:
                    active_indices = np.where(np.argmax(batch_labels, axis=1) != np.argmax(adv_preds, axis=1))[0]
                else:
                    active_indices = np.where(np.argmax(batch_labels, axis=1) == np.argmax(adv_preds, axis=1))[0]

                current_eps += self.eps_step

            adv_x[batch_index_1:batch_index_2] = batch

        return adv_x

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        if isinstance(self.estimator, ClassifierMixin):
            y = check_and_transform_label_format(y, self.estimator.nb_classes)

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = get_labels_np_array(
                    self.estimator.predict(x, batch_size=self.batch_size)  # type: ignore
                )
            y = y / np.sum(y, axis=1, keepdims=True)

            mask = kwargs.get("mask")
            if mask is not None:
                # ensure the mask is broadcastable:
                if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape) :]:
                    raise ValueError("mask shape must be broadcastable to input shape")

            # Return adversarial examples computed with minimal perturbation if option is active
            rate_best: Optional[float]
            if self.minimal:
                logger.info("Performing minimal perturbation FGM.")
                adv_x_best = self._minimal_perturbation(x, y, mask)
                rate_best = 100 * compute_success(
                    self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size,  # type: ignore
                )
            else:
                adv_x_best = None
                rate_best = None

                for _ in range(max(1, self.num_random_init)):
                    adv_x = self._compute(x, x, y, mask, self.eps, self.eps, self._project, self.num_random_init > 0,)

                    if self.num_random_init > 1:
                        rate = 100 * compute_success(
                            self.estimator, x, y, adv_x, self.targeted, batch_size=self.batch_size,  # type: ignore
                        )
                        if rate_best is None or rate > rate_best or adv_x_best is None:
                            rate_best = rate
                            adv_x_best = adv_x
                    else:
                        adv_x_best = adv_x

            logger.info(
                "Success rate of FGM attack: %.2f%%",
                rate_best
                if rate_best is not None
                else 100
                * compute_success(
                    self.estimator,  # type: ignore
                    x,
                    y,
                    adv_x_best,
                    self.targeted,
                    batch_size=self.batch_size,
                ),
            )

        else:
            if self.minimal:
                raise ValueError("Minimal perturbation is only supported for classification.")

            if kwargs.get("mask") is not None:
                raise ValueError("Mask is only supported for classification.")

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = self.estimator.predict(x, batch_size=self.batch_size)

            adv_x_best = self._compute(x, x, y, None, self.eps, self.eps, self._project, self.num_random_init > 0,)

        return adv_x_best

    def _check_params(self) -> None:
        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

        if self.eps <= 0:
            raise ValueError("The perturbation size `eps` has to be positive.")

        if self.eps_step <= 0:
            raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, (int, np.int)):
            raise TypeError("The number of random initialisations has to be of type integer")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.minimal, bool):
            raise ValueError("The flag `minimal` has to be of type bool.")

    def _compute_perturbation(self, batch: np.ndarray, batch_labels: np.ndarray, mask: np.ndarray,optuna_pow:float=1.5) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:
           grad = np.sign(grad)*np.power(abs(grad),optuna_pow)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
        assert batch.shape == grad.shape

        if mask is None:
            return grad
        else:
            return grad * (mask.astype(ART_NUMPY_DTYPE))

    def _apply_perturbation(self, batch: np.ndarray, perturbation: np.ndarray, eps_step: float, perturb_threshold:float) -> np.ndarray:
        for idx in range(batch.shape[0]):
            for i in range(batch.shape[1]): #RGB 3 Channels
                batch_origional = np.copy(batch) #Origional Images: AtLoc (64, 3, 256, 256); DSAC (1,1,480,640)
                batch_origional_plot = batch_origional[idx][i] # 3, 256, 256
                
                grad_pow = np.copy(perturbation) #grad_pow  (64, 3, 256, 256)
                grad_pow_plot = grad_pow[idx][i]# grad_pow_plot:(256, 256)   
 
                if perturb_threshold!=0: 
                    x_adv_range = np.max(batch_origional_plot)-np.min(batch_origional_plot)
                    perturb_threshold = x_adv_range/perturb_threshold
                else:
                    perturb_threshold = 0

                
                #-----------step 1. Take positive top 0.1% grad_pow---------------
                grads_pow_plot_flat_pos = grad_pow_plot.copy().flatten()        
                batch_plot_flat_pos  = batch_origional_plot.copy().flatten()
                
                pos_num=np.sum(grads_pow_plot_flat_pos>0) # the number of positive top 0.1% grad_pow
                
                return_size_top = int(pos_num*0.001)  
                MAX_flat_idx_pos = np.argpartition(grads_pow_plot_flat_pos,-return_size_top)[-return_size_top:] #return indexes of positive top 0.1% grad_pow
                
                
                MAX_value_grad_pow = grads_pow_plot_flat_pos[MAX_flat_idx_pos] #return values of positive top 0.1% grad_pow
                MAX_value_grad_pow_pos_last = np.min(MAX_value_grad_pow) #return the value of the positive top 0.1%-th  grad_pow
                MAX_value_batch = batch_plot_flat_pos[MAX_flat_idx_pos] 
                times_batch_and_pow_pos = MAX_value_batch/MAX_value_grad_pow 
                
                where_are_nan = np.isnan(times_batch_and_pow_pos)
                where_are_inf = np.isinf(times_batch_and_pow_pos)
                times_batch_and_pow_pos[where_are_nan] = 0
                times_batch_and_pow_pos[where_are_inf] = 0
  
                times_batch_and_pow_pos = np.mean(abs(times_batch_and_pow_pos))
                eps_step_pos = times_batch_and_pow_pos
                
                #-----------step 1. end---------------      

                
                #-----------step 2. Take negative top 0.1% grad_pow---------------
                grads_pow_plot_flat_neg = grad_pow_plot.copy().flatten()        
                batch_plot_flat_neg  = batch_origional_plot.copy().flatten()
    
                neg_num=np.sum(grads_pow_plot_flat_pos<0) #the number of negative smallest 0.1% grad_pow 
                return_size_top = int(neg_num*0.001)  #0.05%
                MAX_flat_idx_neg = np.argpartition(grads_pow_plot_flat_neg,return_size_top)[:return_size_top]  #return indexes of negative smallest 0.1% grad_pow
                               
                MAX_value_grad_pow = grads_pow_plot_flat_neg[MAX_flat_idx_neg] #return values of negative smallest 0.1% grad_pow
                MAX_value_grad_pow_neg_last = np.max(MAX_value_grad_pow) #return the value of the negative smallest 0.1%-th grad_pow

                MAX_value_batch = batch_plot_flat_neg[MAX_flat_idx_neg] 
                times_batch_and_pow_neg = MAX_value_batch/MAX_value_grad_pow 
                
                where_are_nan = np.isnan(times_batch_and_pow_neg)
                where_are_inf = np.isinf(times_batch_and_pow_neg)
                times_batch_and_pow_neg[where_are_nan] = 0
                times_batch_and_pow_neg[where_are_inf] = 0

                times_batch_and_pow_neg = np.mean(abs(times_batch_and_pow_neg))
                eps_step_neg = times_batch_and_pow_neg
                #-----------step 2. end---------------

                #-----------step 3. Take median top 50% grad_pow---------------
                grads_pow_plot_flat = grad_pow_plot.copy().flatten()        
                batch_plot_flat  = batch_origional_plot.copy().flatten()
                return_size = int((pos_num)*0.50)
                MAX_flat_idx = np.argpartition(grads_pow_plot_flat,-return_size)[-return_size:] 
                
                MAX_value_grad_pow = grads_pow_plot_flat[MAX_flat_idx]
                MAX_value_batch = batch_plot_flat[MAX_flat_idx]
                times_batch_and_pow = MAX_value_batch/MAX_value_grad_pow 
                
                where_are_nan = np.isnan(times_batch_and_pow)
                where_are_inf = np.isinf(times_batch_and_pow)
                times_batch_and_pow[where_are_nan] = 0
                times_batch_and_pow[where_are_inf] = 0
                times_batch_and_pow = np.mean(abs(times_batch_and_pow))
                eps_step = times_batch_and_pow
                #-----------step 3. end---------------
            
                perturbation[idx][i] = np.where(perturbation[idx][i]>MAX_value_grad_pow_pos_last, eps_step_pos*perturbation[idx][i],np.where(perturbation[idx][i]<MAX_value_grad_pow_neg_last,eps_step_neg*perturbation[idx][i],eps_step* perturbation[idx][i]))
                # perturbation = eps_step * perturbation


                if perturb_threshold ==0:
                    print('perturb_threshold=0, no perturb_threshold!!')
                else:    
                    perturbation[idx][i][np.where(perturbation[idx][i]>perturb_threshold)]=perturb_threshold
                    perturbation[idx][i][np.where(perturbation[idx][i]<-perturb_threshold)]=-perturb_threshold
                #print('================ perturbation[idx][i] min={}, max={}, mean={}',np.max(perturbation[idx][i][0,0,:,:,:]), np.min(perturbation[idx][i][0,0,:,:,:]),np.mean(perturbation[idx][i][0,0,:,:,:]))

                batch[idx][i] = batch[idx][i] +perturbation[idx][i]   #batch is origional images, AtLoc (64, 3, 256, 256)
                batch[idx][i][np.where(batch[idx][i]>1)]=1
                batch[idx][i][np.where(batch[idx][i]<0)]=0
                #print('================ batch_ADD_ADV min={}, max={}, mean={}',np.max(batch[idx][i][0,0,:,:,:]), np.min(batch[idx][i][0,0,:,:,:]),np.mean(batch[idx][i][0,0,:,:,:]))
                
        
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            batch = np.clip(batch, clip_min, clip_max)

        return batch


    def _compute(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        eps: float, #Perturb step size=0.3
        eps_step: float, #Perturb step size
        project: bool,
        random_init: bool,
        perturb_threshold: int=1,
        optuna_pow:float=1.5,
    ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            x_adv = x.astype(ART_NUMPY_DTYPE)



            # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels, mask_batch,optuna_pow)

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step,perturb_threshold)

            if project:
                perturbation = projection(
                    x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], eps, self.norm
                )
                x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv


    def generate_mapnet(self, x: np.ndarray, y: np.ndarray,perturb_threshold:int=1,optuna_eps:float=158,optuna_pow:float=1.5,optuna_on:bool=False, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """



        if isinstance(self.estimator, ClassifierMixin):
            if not isinstance(x, np.ndarray):
                x = x.np() 
            if not isinstance(y, np.ndarray):
                y = y.np() 

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = get_labels_np_array(
                    self.estimator.predict_mapnet(x,y, batch_size=self.batch_size)  # type: ignore
                )

            mask = kwargs.get("mask")	
            if mask is not None:
                
                # ensure the mask is broadcastable:
                if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape) :]:
                    raise ValueError("mask shape must be broadcastable to input shape")

            # Return adversarial examples computed with minimal perturbation if option is active
            rate_best: Optional[float]
            if self.minimal:
                
                logger.info("Performing minimal perturbation FGM.")
                adv_x_best = self._minimal_perturbation(x, y, mask)
                rate_best = 100 * compute_success(
                    self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size,  # type: ignore
                )
            else:
                
                adv_x_best = None
                rate_best = None
                
                
                for _ in range(max(1, self.num_random_init)):
                    adv_x = self._compute(x, x, y, mask, optuna_eps,optuna_eps , self._project, self.num_random_init > 0,perturb_threshold,optuna_pow)

                    if self.num_random_init > 1:
                        
                        rate = 100 * compute_success(
                            self.estimator, x, y, adv_x, self.targeted, batch_size=self.batch_size,  # type: ignore
                        )
                        if rate_best is None or rate > rate_best or adv_x_best is None:
                            
                            rate_best = rate
                            adv_x_best = adv_x
                    else:
                        adv_x_best = adv_x


                logger.info(
                "Success rate of mapnet FGM attack: %.2f%%",
                rate_best
                if rate_best is not None
                else 100
                * 1.000,
            )

        else:
            if self.minimal:
                raise ValueError("Minimal perturbation is only supported for classification.")

            if kwargs.get("mask") is not None:
                raise ValueError("Mask is only supported for classification.")

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = self.estimator.predict(x, batch_size=self.batch_size)

            adv_x_best = self._compute(x, x, y, None, self.eps, self.eps, self._project, self.num_random_init > 0,)

        return adv_x_best



    def generate_ransac(self, x: np.ndarray, y: np.ndarray, ransac_grad:float ,perturb_threshold: int ,optuna_eps:float ,optuna_pow:float ,optuna_on:bool=False, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """


        if isinstance(self.estimator, ClassifierMixin):
            
            
            if not isinstance(x, np.ndarray):
                x = x.np() 
            if not isinstance(y, np.ndarray):
                y = y.np() 

            if y is None:
                
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = get_labels_np_array(
                    self.estimator.predict_mapnet(x,y, batch_size=self.batch_size)  # type: ignore
                )
            # y = y / np.sum(y, axis=1, keepdims=True)

            mask = kwargs.get("mask")
            if mask is not None:
                
                # ensure the mask is broadcastable:
                if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape) :]:
                    raise ValueError("mask shape must be broadcastable to input shape")

            # Return adversarial examples computed with minimal perturbation if option is active
            rate_best: Optional[float]
            if self.minimal:
                
                logger.info("Performing minimal perturbation FGM.")
                adv_x_best = self._minimal_perturbation(x, y, mask)
                rate_best = 100 * compute_success(
                    self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size,  # type: ignore
                )
            else:
                
                adv_x_best = None
                rate_best = None
                
                for _ in range(max(1, self.num_random_init)):
                    adv_x = self._compute_ransac(x, x, y, mask, optuna_eps,optuna_eps , self._project, self.num_random_init > 0,perturb_threshold,optuna_pow, ransac_grad)

                    if self.num_random_init > 1:
                        
                        rate = 100 * compute_success(
                            self.estimator, x, y, adv_x, self.targeted, batch_size=self.batch_size,  # type: ignore
                        )
                        if rate_best is None or rate > rate_best or adv_x_best is None:
                            
                            rate_best = rate
                            adv_x_best = adv_x
                    else:
                        adv_x_best = adv_x


                logger.info(
                "Success rate of mapnet FGM attack: %.2f%%",
                rate_best
                if rate_best is not None
                else 100
                * 1.000,
            )

        else:
            if self.minimal:
                raise ValueError("Minimal perturbation is only supported for classification.")

            if kwargs.get("mask") is not None:
                raise ValueError("Mask is only supported for classification.")

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = self.estimator.predict(x, batch_size=self.batch_size)

            adv_x_best = self._compute(x, x, y, None, self.eps, self.eps, self._project, self.num_random_init > 0,)

        return adv_x_best

    def _compute_ransac(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        eps: float, #Perturb step size=0.3
        eps_step: float, #Perturb step size
        project: bool,
        random_init: bool,
        perturb_threshold: int=10,
        optuna_pow:float=1.5,
        ransac_grad:float=1.0,
    ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            x_adv = x.astype(ART_NUMPY_DTYPE)

   

            # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]
            # Get perturbation
            perturbation = self._compute_perturbation_ransac(batch, batch_labels, mask_batch,optuna_pow, ransac_grad)

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step,perturb_threshold)

            if project:
                perturbation = projection(
                    x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], eps, self.norm
                )
                x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv

    def _compute_perturbation_ransac(self, batch: np.ndarray, batch_labels: np.ndarray, mask: np.ndarray,optuna_pow:float=1.5,ransac_grad:float=1.0) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient_ransac(batch, batch_labels, ransac_grad) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:           
           grad = np.sign(grad)*np.power(abs(grad),optuna_pow)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
        assert batch.shape == grad.shape

        if mask is None:
            return grad
        else:
            return grad * (mask.astype(ART_NUMPY_DTYPE))



    def generate_dsac_init(self, x: np.ndarray, y: np.ndarray,opt_mode, pixel_grid, cam_mat, opt_mindepth, opt_hardclamp, use_init, opt_softclamp, opt_targetdepth, focal_length,perturb_threshold,optuna_eps,optuna_pow,optuna_on:bool=False, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """


        if isinstance(self.estimator, ClassifierMixin):
            
            
            if not isinstance(x, np.ndarray):
                x = x.np() 
            if not isinstance(y, np.ndarray):
                y = y.np() 

            if y is None:
                
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = get_labels_np_array(
                    self.estimator.predict_mapnet(x,y, batch_size=self.batch_size)  # type: ignore
                )
            # y = y / np.sum(y, axis=1, keepdims=True)

            mask = kwargs.get("mask")
            if mask is not None:
                
                # ensure the mask is broadcastable:
                if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape) :]:
                    raise ValueError("mask shape must be broadcastable to input shape")

            # Return adversarial examples computed with minimal perturbation if option is active
            rate_best: Optional[float]
            if self.minimal:
                
                logger.info("Performing minimal perturbation FGM.")
                adv_x_best = self._minimal_perturbation(x, y, mask)
                rate_best = 100 * compute_success(
                    self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size,  # type: ignore
                )
            else:
                
                adv_x_best = None
                rate_best = None
                
                for _ in range(max(1, self.num_random_init)):
                    adv_x = self._compute_dsac_init(x, x, y, mask, optuna_eps,optuna_eps , self._project, self.num_random_init > 0,opt_mode, pixel_grid, cam_mat, opt_mindepth, opt_hardclamp, use_init, opt_softclamp, opt_targetdepth, focal_length,perturb_threshold,optuna_pow)

                    if self.num_random_init > 1:
                        
                        rate = 100 * compute_success(
                            self.estimator, x, y, adv_x, self.targeted, batch_size=self.batch_size,  # type: ignore
                        )
                        if rate_best is None or rate > rate_best or adv_x_best is None:
                            
                            rate_best = rate
                            adv_x_best = adv_x
                    else:
                        adv_x_best = adv_x


                logger.info(
                "Success rate of mapnet FGM attack: %.2f%%",
                rate_best
                if rate_best is not None
                else 100
                * 1.000,
            )

        else:
            
            if self.minimal:
                raise ValueError("Minimal perturbation is only supported for classification.")

            if kwargs.get("mask") is not None:
                raise ValueError("Mask is only supported for classification.")

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = self.estimator.predict(x, batch_size=self.batch_size)

            adv_x_best = self._compute(x, x, y, None, self.eps, self.eps, self._project, self.num_random_init > 0,)

        return adv_x_best

    def _compute_dsac_init(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        eps: float, #Perturb step size=0.3
        eps_step: float, #Perturb step size
        project: bool,
        random_init: bool, opt_mode, pixel_grid, cam_mat, opt_mindepth, opt_hardclamp, use_init, opt_softclamp, opt_targetdepth, focal_length,perturb_threshold: int=10, optuna_pow:float=1.5
    ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            x_adv = x.astype(ART_NUMPY_DTYPE)

            # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation_dsac_init(batch, batch_labels, mask_batch, opt_mode, pixel_grid, cam_mat, opt_mindepth, opt_hardclamp, use_init, opt_softclamp, opt_targetdepth, focal_length, optuna_pow)

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step,perturb_threshold)

            if project:
                perturbation = projection(
                    x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], eps, self.norm
                )
                x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv

    def _compute_perturbation_dsac_init(self, batch: np.ndarray, batch_labels: np.ndarray, mask: np.ndarray, opt_mode, pixel_grid, cam_mat, opt_mindepth, opt_hardclamp, use_init, opt_softclamp, opt_targetdepth, focal_length, optuna_pow:float=1.5) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient_dsac_init(batch, batch_labels, opt_mode, pixel_grid, cam_mat, opt_mindepth, opt_hardclamp, use_init, opt_softclamp, opt_targetdepth, focal_length) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == np.inf:
           grad = np.sign(grad)*np.power(abs(grad),optuna_pow)
           # grad = np.sign(grad)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
        assert batch.shape == grad.shape

        if mask is None:
            return grad
        else:
            return grad * (mask.astype(ART_NUMPY_DTYPE))
