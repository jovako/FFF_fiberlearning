from math import prod

import torch

from fff.base import FreeFormBaseHParams, FreeFormBase, VolumeChangeResult
from fff.utils.truncate import Truncate
from fff.utils.subject_model import SubjectModel


class FreeFormInjectiveFlowHParams(FreeFormBaseHParams):
    pass


class FreeFormInjectiveFlow(FreeFormBase):
    """
    A FreeFormInjectiveFlow is an injective flow consisting of a pair of free-form
    encoder and decoder.
    """
    hparams: FreeFormInjectiveFlowHParams

    def __init__(self, hparams: FreeFormInjectiveFlowHParams | dict):
        if not isinstance(hparams, FreeFormInjectiveFlowHParams):
            hparams = FreeFormInjectiveFlowHParams(**hparams)
        super().__init__(hparams)
        load_sm = hparams["load_subject_model"]
        if load_sm:
            print("loading subject_model")
            sm_dir = hparams["data_set"]["root"]
            subject_model = FreeFormInjectiveFlow.load_from_checkpoint(
                f"{sm_dir}/subject_model/checkpoints/best.ckpt"
            )
            subject_model.eval()
            for param in subject_model.parameters():
                param.require_grad = False
            self.subject_model = SubjectModel(subject_model, subject_model.encode)
            #self.subject_model = Truncate(Classifier)
            """
            else:
                print("subject_model is Autoencoder")
                if hparams["data_set"]["path"]=="fif_moons":
                    self.subject_model = FreeFormInjectiveFlow.load_from_checkpoint(
                            "subject_models/moons_FIF/checkpoints/last.ckpt"
                    )
                elif hparams["data_set"]["path"] in ["16EMnist_F3F"]:
                    self.subject_model = FreeFormInjectiveFlow.load_from_checkpoint(
                            "subject_models/16EMnist_F3F/checkpoints/last.ckpt"
                    )
                elif hparams["data_set"]["path"] in ["16EMnist_F5F"]:
                    self.subject_model = FreeFormInjectiveFlow.load_from_checkpoint(
                            "subject_models/16EMnist_F5F/checkpoints/last.ckpt"
                    )
                elif hparams["data_set"]["path"] in ["16EMnist_F3F_4"]:
                    self.subject_model = FreeFormInjectiveFlow.load_from_checkpoint(
                            "subject_models/16EMnist_F3F_4/checkpoints/last.ckpt"
                    )
            """
        if self.data_dim <= self.latent_dim:
            raise ValueError("Latent dimension must be less than data dimension "
                             "for a FreeFormInjectiveFlow.")

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        z, jac_enc = self._encoder_jac(x, c, **kwargs)
        jac_enc = jac_enc.reshape(x.shape[0], prod(z.shape[1:]), prod(x.shape[1:]))
        jtj = torch.einsum("bik,bjk->bij", jac_enc, jac_enc)
        log_det = jtj.slogdet()[1] / 2
        return VolumeChangeResult(z, log_det, {})

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        # Forward gradient is faster because latent dimension is smaller than data dimension
        x1, jac_dec = self._decoder_jac(z, c, grad_type="forward", **kwargs)
        jac_dec = jac_dec.reshape(z.shape[0], prod(x1.shape[1:]), prod(z.shape[1:]))
        jjt = torch.einsum("bki,bkj->bij", jac_dec, jac_dec)
        log_det = jjt.slogdet()[1] / 2
        return VolumeChangeResult(x1, log_det, {})
