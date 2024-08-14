# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from timeit import default_timer as timer

import numpy as np
import torch
import torch.optim as optim
from splitnn.mri_dataset import MRI_DATASET
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.workflows.splitnn_workflow import SplitNNConstants, SplitNNDataKind
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.utils import fobs
from sklearn import metrics
import matplotlib.pyplot as plt
import io
from PIL import Image


class CIFAR10LearnerSplitNN(Learner):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        intersection_file: str = None,
        lr: float = 1e-3,
        model: dict = None,
        analytic_sender_id: str = "analytic_sender",
        fp16: bool = True,
        val_freq: int = 3470,
    ):
        """Simple CIFAR-10 Trainer for split learning.

        Args:
            dataset_root: directory with CIFAR-10 data.
            intersection_file: Optional. intersection file specifying overlapping indices between both clients.
                Defaults to `None`, i.e. the whole training dataset is used.
            lr: learning rate.
            model: Split learning model.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            fp16: If `True`, convert activations and gradients send between clients to `torch.float16`.
                Reduces bandwidth needed for communication but might impact model accuracy.
            val_freq: how often to perform validation in rounds. Defaults to 1000. No validation if <= 0.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.intersection_file = intersection_file
        self.lr = lr
        self.model = model
        self.analytic_sender_id = analytic_sender_id
        self.fp16 = fp16
        self.val_freq = val_freq

        self.target_names = None
        self.app_root = None
        self.current_round = None
        self.num_rounds = None
        self.batch_size = None
        self.writer = None
        self.client_name = None
        self.other_client = None
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        self.split_id = None
        self.train_activations = None
        self.train_batch_indices = None
        self.train_size = 0
        self.val_loss = []
        self.val_labels = []
        self.val_pred_labels = []
        self.compute_stats_pool = None
        self.TP, self.FP, self.FN, self.TN = 0, 0, 0, 0
        self.all_outputs = []
        self.all_labels = []
        self.batch_indices = []

        # use FOBS serializing/deserializing PyTorch tensors
        fobs.register(TensorDecomposer)

    def _get_model(self, fl_ctx: FLContext):
        """Get model from client config. Modelled after `PTFileModelPersistor`."""
        if isinstance(self.model, str):
            # treat it as model component ID
            model_component_id = self.model
            engine = fl_ctx.get_engine()
            self.model = engine.get_component(model_component_id)
            if not self.model:
                self.log_error(fl_ctx, f"cannot find model component '{model_component_id}'")
                return
        if self.model and isinstance(self.model, dict):
            # try building the model
            try:
                engine = fl_ctx.get_engine()
                # use provided or default optimizer arguments and add the model parameters
                if "args" not in self.model:
                    self.model["args"] = {}
                self.model = engine.build_component(self.model)
            except Exception as e:
                self.system_panic(
                    f"Exception while parsing `model`: " f"{self.model} with Exception {e}",
                    fl_ctx,
                )
                return
        if self.model and not isinstance(self.model, torch.nn.Module):
            self.system_panic(f"expect model to be torch.nn.Module but got {type(self.model)}: {self.model}", fl_ctx)
            return
        if self.model is None:
            self.system_panic(f"Model wasn't built correctly! It is {self.model}", fl_ctx)
            return
        self.log_info(fl_ctx, f"Running model {self.model}")

    def initialize(self, parts: dict, fl_ctx: FLContext):
        t_start = timer()
        self._get_model(fl_ctx=fl_ctx)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.BCELoss()



        self.transform_train = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform_valid = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.client_name = fl_ctx.get_identity_name()
        self.split_id = self.model.get_split_id()
        self.log_info(fl_ctx, f"Running `split_id` {self.split_id} on site `{self.client_name}`")

        if self.split_id == 0:  # data side
            data_returns = "image"
        elif self.split_id == 1:  # label side
            data_returns = "label"
        else:
            raise ValueError(f"Expected split_id to be '0' or '1' but was {self.split_id}")

        if self.intersection_file is not None:
            _intersect_indices = np.loadtxt(self.intersection_file)
        else:
            _intersect_indices = None
        self.train_dataset = MRI_DATASET(
            train=True,
            returns=data_returns,
            intersect_idx=None,
        )

        self.valid_dataset = MRI_DATASET(
            train=False,
            returns=data_returns,
            intersect_idx=None,  # TODO: support validation intersect indices
        )
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")

        self.train_size = len(self.train_dataset)
        if self.train_size <= 0:
            raise ValueError(f"Expected train dataset size to be larger zero but got {self.train_size}")
        self.log_info(fl_ctx, f"Training with {self.train_size} overlapping indices of {self.train_dataset.orig_size}.")

        # Select local TensorBoard writer or event-based writer for streaming
        if self.split_id == 1:  # metrics can only be computed for client with labels
            self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
            if not self.writer:  # use local TensorBoard writer only
                self.writer = SummaryWriter(self.app_root)

        # register aux message handlers
        engine = fl_ctx.get_engine()

        if self.split_id == 1:
            engine.register_aux_message_handler(
                topic=SplitNNConstants.TASK_TRAIN_LABEL_STEP, message_handle_func=self._aux_train_label_side
            )
            engine.register_aux_message_handler(
                topic=SplitNNConstants.TASK_VALID_LABEL_STEP, message_handle_func=self._aux_val_label_side
            )
            self.log_debug(fl_ctx, f"Registered aux message handlers for split_id {self.split_id}")

        self.compute_stats_pool = StatsPoolManager.add_time_hist_pool(
            "Compute_Time", "Compute time in secs", scope=self.client_name
        )

        self.compute_stats_pool.record_value(category="initialize", value=timer() - t_start)

    """ training steps """

    def _train_step_data_side(self, batch_indices):
        t_start = timer()
        self.model.train()

        inputs = self.train_dataset.get_batch(batch_indices)
        #inputs = inputs.unsqueeze(2)
        inputs = inputs.float()
        inputs = inputs.to(self.device)

        self.train_activations = self.model.forward(inputs)  # keep on site-1

        self.compute_stats_pool.record_value(category="_train_step_data_side", value=timer() - t_start)

        return self.train_activations.detach().requires_grad_()  # x to be sent to other client

    def _val_step_data_side(self, batch_indices):
        t_start = timer()
        self.model.eval()
        with torch.no_grad():

            inputs = self.valid_dataset.get_batch(batch_indices)
        #inputs = inputs.unsqueeze(2)
            inputs = inputs.float()
            inputs = inputs.to(self.device)

            _val_activations = self.model.forward(inputs)  # keep on site-1

        self.compute_stats_pool.record_value(category="_val_step_data_side", value=timer() - t_start)

        return _val_activations.detach()#.flatten(start_dim=1, end_dim=-1)  # x to be sent to other client

    def _train_step_label_side(self, batch_indices, activations, fl_ctx: FLContext):
        t_start = timer()

        self.model.train()
        self.optimizer.zero_grad()

        labels = self.train_dataset.get_batch(batch_indices)
        labels = labels.to(self.device)

        if self.fp16:
            activations = activations.type(torch.float32)  # return to default pytorch precision

        activations = activations.to(self.device)
        activations.requires_grad_(True)

        pred = self.model.forward(activations)
        pred = torch.sigmoid(pred)
        #print(pred.size(),labels.size())
        #print(pred.squeeze(),pred)
        loss = self.criterion(pred.squeeze(), labels.float())
        loss.backward()

        pred_labels = pred > 0.5
        acc = (pred_labels == labels.unsqueeze(1)).sum() / len(labels)

        if self.current_round % 100 == 0:
            self.log_info(
                fl_ctx,
                f"Round {self.current_round}/{self.num_rounds} train_loss: {loss.item():.4f}, train_accuracy: {acc.item():.4f}",
            )
        if self.writer:
            self.writer.add_scalar("train_loss", loss, self.current_round)
            self.writer.add_scalar("train_accuracy", acc, self.current_round)

        self.optimizer.step()

        self.compute_stats_pool.record_value(category="_train_step_label_side", value=timer() - t_start)

        if not isinstance(activations.grad, torch.Tensor):
            raise ValueError("No valid gradients available!")
        # gradient to be returned to other client
        if self.fp16:
            return activations.grad.type(torch.float16)
        else:
            return activations.grad

    def _val_step_label_side(self, batch_indices, activations, fl_ctx: FLContext):
        t_start = timer()
        self.model.eval()
        with torch.no_grad():
            labels = self.valid_dataset.get_batch(batch_indices)
            labels = labels.to(self.device)
            self.batch_indices.append(batch_indices)

            if self.fp16:
                activations = activations.type(torch.float32)  # return to default pytorch precision

            activations = activations.to(self.device)


            pred = self.model.forward(activations)
            #print(pred,labels)
            pred = torch.sigmoid(pred)
            #print(pred)
            loss = self.criterion(pred.squeeze(), labels.float())
            self.val_loss.append(loss.unsqueeze(0))  # unsqueeze needed for later concatenation

            pred_labels = pred > 0.5

            self.val_pred_labels.append(pred)#.unsqueeze(0)
            self.val_labels.append(labels)#.unsqueeze(0)

            self.compute_stats_pool.record_value(category="_val_step_label_side", value=timer() - t_start)

    def _log_validation(self, fl_ctx: FLContext):
        if len(self.val_loss) > 0:
            loss = torch.mean(torch.cat(self.val_loss))

            _val_pred_labels = torch.cat(self.val_pred_labels).cpu().numpy()
            _val_labels = torch.cat(self.val_labels).cpu().numpy()
            #print(_val_pred_labels,_val_labels)

            acc = (_val_pred_labels == _val_labels).sum() / len(_val_labels)
            #print(acc, len(_val_labels))

            # Berechnung von TP, TN, FP, FN (keine wirkliche Aussagekraft)
            self.TP = ((_val_pred_labels == 1) & (_val_labels == 1)).sum().item()
            self.TN = ((_val_pred_labels == 0) & (_val_labels == 0)).sum().item()
            self.FP = ((_val_pred_labels == 1) & (_val_labels == 0)).sum().item()
            self.FN = ((_val_pred_labels == 0) & (_val_labels == 1)).sum().item()
            x = str(_val_pred_labels) + str(_val_labels)
            #print(self.TP, self.TN, self.FP, self.FN)


            fpr, tpr, thresholds = metrics.roc_curve(_val_labels, _val_pred_labels)
            roc_auc = metrics.auc(fpr, tpr)

            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')


            """
            self.writer.add_figure("Fig1",plt)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            # Konvertieren des Pufferinhalts in ein NumPy-Array
            image = np.array(Image.open(buf))

            # Speichern des Bildes mit dem TensorBoard-Schreiber
            self.writer.add_image(f'ROC Curve_{self.current_round}', image, global_step=self.current_round)
            """



            self.log_info(
                fl_ctx,
                f"Round {self.current_round}/{self.num_rounds} val_loss: {loss.item():.4f}, val_accuracy: {acc.item():.4f}",
            )
            if self.writer:
                self.writer.add_scalar("val_loss", loss, self.current_round)
                self.writer.add_scalar("val_accuracy", acc, self.current_round)
                self.writer.add_scalar("val_TP", self.TP, self.current_round)
                self.writer.add_scalar("val_TN", self.TN, self.current_round)
                self.writer.add_scalar("val_FP", self.FP, self.current_round)
                self.writer.add_scalar("val_FN", self.FN, self.current_round)
                self.writer.add_figure(f'ROC Curve_{self.current_round}', plt.gcf(),self.current_round)
                self.writer.add_text("val_labels", x,self.current_round)



            self.batch_indices= []
            self.val_loss = []
            self.val_labels = []
            self.val_pred_labels = []
            self.TP, self.FP, self.FN, self.TN = 0, 0, 0, 0


    def _backward_step_data_side(self, gradient, fl_ctx: FLContext):
        t_start = timer()
        self.model.train()
        self.optimizer.zero_grad()

        if self.fp16:
            gradient = gradient.type(torch.float32)  # return to default pytorch precision

        gradient = gradient.to(self.device)
        self.train_activations.backward(gradient=gradient.reshape(self.train_activations.shape))
        self.optimizer.step()

        self.log_debug(
            fl_ctx, f"{self.client_name} runs model with `split_id` {self.split_id} for backward step on data side."
        )
        self.compute_stats_pool.record_value(category="_backward_step_data_side", value=timer() - t_start)

    def _train_forward_backward_data_side(self, fl_ctx: FLContext, gradient=None) -> Shareable:
        t_start = timer()
        # combine forward and backward on data client
        # 1. perform backward step if gradients provided
        if gradient is not None:
            result_backward = self._backward_data_side(gradient, fl_ctx=fl_ctx)
            assert (
                result_backward.get_return_code() == ReturnCode.OK
            ), f"Backward step failed with return code {result_backward.get_return_code()}"
        # 2. compute activations
        activations = self._train_data_side(fl_ctx=fl_ctx)

        self.compute_stats_pool.record_value(category="_train_forward_backward_data_side", value=timer() - t_start)

        return activations#.flatten(start_dim=1, end_dim=-1)  # keep batch dim

    def _train_data_side(self, fl_ctx: FLContext) -> Shareable:
        t_start = timer()
        if self.split_id != 0:
            raise ValueError(
                f"Expected `split_id` 0. It doesn't make sense to run `_train_data_side` with `split_id` {self.split_id}"
            )

        self.log_debug(fl_ctx, f"Train data side in round {self.current_round} of {self.num_rounds} rounds.")

        act = self._train_step_data_side(batch_indices=self.train_batch_indices)

        self.log_debug(
            fl_ctx, f"{self.client_name} finished model with `split_id` {self.split_id} for train on data side."
        )

        self.compute_stats_pool.record_value(category="_train_data_side", value=timer() - t_start)

        self.log_debug(fl_ctx, f"Sending train data activations: {type(act)}")

        if self.fp16:
            return act.type(torch.float16)
        else:
            return act

    def _aux_train_label_side(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """train aux message handler"""
        t_start = timer()
        if self.split_id != 1:
            raise ValueError(
                f"Expected `split_id` 1. It doesn't make sense to run `_aux_train_label_side` with `split_id` {self.split_id}"
            )

        self.current_round = request.get_header(AppConstants.CURRENT_ROUND)
        self.num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_debug(fl_ctx, f"Train label in round {self.current_round} of {self.num_rounds} rounds.")

        dxo = from_shareable(request)
        if dxo.data_kind != SplitNNDataKind.ACTIVATIONS:
            raise ValueError(f"Expected data kind {SplitNNDataKind.ACTIVATIONS} but received {dxo.data_kind}")

        batch_indices = dxo.get_meta_prop(SplitNNConstants.BATCH_INDICES)
        if batch_indices is None:
            raise ValueError("No batch indices in DXO!")

        activations = dxo.data.get(SplitNNConstants.DATA)
        if activations is None:
            raise ValueError("No activations in DXO!")

        gradient = self._train_step_label_side(
            batch_indices=batch_indices, activations=fobs.loads(activations), fl_ctx=fl_ctx
        )

        self.log_debug(fl_ctx, "_aux_train_label_side finished.")
        return_shareable = DXO(
            data={SplitNNConstants.DATA: fobs.dumps(gradient)}, data_kind=SplitNNDataKind.GRADIENT
        ).to_shareable()

        self.compute_stats_pool.record_value(category="_aux_train_label_side", value=timer() - t_start)

        self.log_debug(fl_ctx, f"Sending train label return_shareable: {type(return_shareable)}")
        return return_shareable

    def _aux_val_label_side(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """validation aux message handler"""
        t_start = timer()
        if self.split_id != 1:
            raise ValueError(
                f"Expected `split_id` 1. It doesn't make sense to run `_aux_train_label_side` with `split_id` {self.split_id}"
            )

        val_round = request.get_header(AppConstants.CURRENT_ROUND)
        val_num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_debug(fl_ctx, f"Validate label in round {self.current_round} of {self.num_rounds} rounds.")

        dxo = from_shareable(request)
        if dxo.data_kind != SplitNNDataKind.ACTIVATIONS:
            raise ValueError(f"Expected data kind {SplitNNDataKind.ACTIVATIONS} but received {dxo.data_kind}")

        batch_indices = dxo.get_meta_prop(SplitNNConstants.BATCH_INDICES)
        if batch_indices is None:
            raise ValueError("No batch indices in DXO!")

        activations = dxo.data.get(SplitNNConstants.DATA)
        if activations is None:
            raise ValueError("No activations in DXO!")

        self._val_step_label_side(batch_indices=batch_indices, activations=fobs.loads(activations), fl_ctx=fl_ctx)

        if val_round == val_num_rounds - 1:
            self._log_validation(fl_ctx)

        self.compute_stats_pool.record_value(category="_aux_val_label_side", value=timer() - t_start)

        return make_reply(ReturnCode.OK)

    def _backward_data_side(self, gradient, fl_ctx: FLContext) -> Shareable:
        t_start = timer()
        if self.split_id != 0:
            raise ValueError(
                f"Expected `split_id` 0. It doesn't make sense to run `_backward_data_side` with `split_id` {self.split_id}"
            )

        self._backward_step_data_side(gradient=fobs.loads(gradient), fl_ctx=fl_ctx)

        self.log_debug(fl_ctx, "_backward_data_side finished.")

        self.compute_stats_pool.record_value(category="_backward_data_side", value=timer() - t_start)

        return make_reply(ReturnCode.OK)

    # Model initialization task (one time only in beginning)
    def init_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        t_start = timer()
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        '''for var_name in local_var_dict:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed.") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError("No global weights loaded!")
        '''
        self.compute_stats_pool.record_value(category="init_model", value=timer() - t_start)

        self.log_info(fl_ctx, "init_model finished.")

        return make_reply(ReturnCode.OK)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        t_start = timer()
        """main training logic"""
        engine = fl_ctx.get_engine()

        self.num_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        if not self.num_rounds:
            raise ValueError("No number of rounds available.")
        self.batch_size = shareable.get_header(SplitNNConstants.BATCH_SIZE)
        self.target_names = np.asarray(
            shareable.get_header(SplitNNConstants.TARGET_NAMES)
        )  # convert to array for string matching below
        self.other_client = self.target_names[self.target_names != self.client_name][0]
        self.log_info(fl_ctx, f"Starting training of {self.num_rounds} rounds with batch size {self.batch_size}")

        self.train_indices = []
        gradients = None  # initial gradients
        for _curr_round in range(self.num_rounds):
            self.current_round = _curr_round
            if self.split_id != 0:
                continue  # only run this logic on first site
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            self.log_debug(fl_ctx, f"Starting current round={self.current_round} of {self.num_rounds}.")
            if len(self.train_indices)< self.batch_size:
                train_indices_new = np.arange(self.train_size)
                np.random.shuffle(train_indices_new)
                self.train_indices = np.concatenate((self.train_indices, train_indices_new))
            self.train_batch_indices = self.train_indices[:self.batch_size]
            self.train_indices = self.train_indices[self.batch_size:]
            #np.random.randint(0, self.train_size - 1, self.batch_size)

            # Site-1 image forward & backward (from 2nd round)
            fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=False)
            activations = self._train_forward_backward_data_side(fl_ctx, gradients)

            # Site-2 label loss & backward
            dxo = DXO(data={SplitNNConstants.DATA: fobs.dumps(activations)}, data_kind=SplitNNDataKind.ACTIVATIONS)
            dxo.set_meta_prop(SplitNNConstants.BATCH_INDICES, self.train_batch_indices)

            data_shareable = dxo.to_shareable()
            data_shareable.set_header(AppConstants.CURRENT_ROUND, self.current_round)
            data_shareable.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
            data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self.current_round)

            # send to other side
            result = engine.send_aux_request(
                targets=self.other_client,
                topic=SplitNNConstants.TASK_TRAIN_LABEL_STEP,
                request=data_shareable,
                timeout=SplitNNConstants.TIMEOUT,
                fl_ctx=fl_ctx,
            )
            shareable = result.get(self.other_client)
            if shareable is not None:
                dxo = from_shareable(shareable)
                if dxo.data_kind != SplitNNDataKind.GRADIENT:
                    raise ValueError(f"Expected data kind {SplitNNDataKind.GRADIENT} but received {dxo.data_kind}")
                gradients = dxo.data.get(SplitNNConstants.DATA)
            else:
                raise ValueError(f"No message returned from {self.other_client}!")

            self.log_debug(fl_ctx, f"Ending current round={self.current_round}.")

            if self.val_freq > 0:
                if _curr_round % self.val_freq == 0:
                    self._validate(fl_ctx)

        self.compute_stats_pool.record_value(category="train", value=timer() - t_start)

        return make_reply(ReturnCode.OK)

    def _validate(self, fl_ctx: FLContext):
        t_start = timer()
        engine = fl_ctx.get_engine()

        idx = np.arange(len(self.valid_dataset))
        n_batches = int(np.ceil(len(self.valid_dataset) / self.batch_size))
        for _val_round, _val_batch_indices in enumerate(np.array_split(idx, n_batches)):
            activations = self._val_step_data_side(batch_indices=_val_batch_indices)

            # Site-2 label loss & accuracy
            dxo = DXO(data={SplitNNConstants.DATA: fobs.dumps(activations)}, data_kind=SplitNNDataKind.ACTIVATIONS)
            dxo.set_meta_prop(SplitNNConstants.BATCH_INDICES, _val_batch_indices)

            data_shareable = dxo.to_shareable()
            data_shareable.set_header(AppConstants.CURRENT_ROUND, _val_round)
            data_shareable.set_header(AppConstants.NUM_ROUNDS, n_batches)
            data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, _val_round)

            # send to other side to validate
            engine.send_aux_request(
                targets=self.other_client,
                topic=SplitNNConstants.TASK_VALID_LABEL_STEP,
                request=data_shareable,
                timeout=SplitNNConstants.TIMEOUT,
                fl_ctx=fl_ctx,
            )

        self.compute_stats_pool.record_value(category="_validate", value=timer() - t_start)

        self.log_debug(fl_ctx, "finished validation.")
