import numpy as np
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    SkyModel,
    #PowerLawNuisanceSpectralModel,
    #PowerLawNormNuisanceSpectralModel,
    #ExpCutoffPowerLawNuisanceSpectralModel,
    GaussianSpatialModel)
#from MapDatasetNuisanceE import MapDatasetNuisanceE
from gammapy.modeling import Parameter, Parameters
from gammapy.datasets import MapDataset

from gammapy.modeling.models import SpectralModel
from gammapy.modeling.models.cube import IRFModel

path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'


#ExpCutoffPowerLaw instead of Powerlaw

class sys_dataset():
    def __init__(self, 
                 dataset_asimov,
                 shift,
                 tilt,
                 rnd,
                 amplitude,
                 sigma):
        self.dataset_asimov = dataset_asimov
        self.shift = shift
        self.tilt = tilt
        self.rnd = rnd
        self.amplitude = amplitude
        self.sigma = sigma
        
    def set_model(self):
        model_spatial = GaussianSpatialModel(
            lon_0="83.631 deg",
            lat_0="22.018 deg",
            sigma= f"{self.sigma} deg"
        )
        model_spectrum = ExpCutoffPowerLawSpectralModel(
            index=2.3,
            amplitude=f"{self.amplitude} TeV-1 cm-2 s-1", 
            lambda_=  "0.1 TeV-1"
        )
        source_model = SkyModel(spatial_model = model_spatial,
                               spectral_model = model_spectrum,
                               name = "Source")    
        source_model.parameters['lon_0'].frozen = True
        source_model.parameters['lat_0'].frozen = True
        source_model.parameters['sigma'].frozen = True
        models = Models(source_model)
        return models
    
    def create_dataset(self):
        dataset = self.dataset_asimov.copy()
        models = self.set_model()
        #bkg model
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        dataset.models = models
        
        if self.rnd:
            counts_data = np.random.poisson(dataset.npred().data)
        else:
            counts_data = dataset.npred().data

        dataset.counts.data = counts_data
        
        #irf model
        IRFmodel = IRFModel(dataset_name = dataset.name)
        IRFmodel.parameters['tilt_nuisance'].frozen  = False
        models.append(IRFmodel)
        
        dataset.models = models
        dataset.models.parameters['norm_nuisance'].value  = self.shift
        dataset.models.parameters['tilt_nuisance'].value  = self.tilt
        dataset.exposure = dataset.npred_exposure()
        
        # set models without the IRF model
        models = self.set_model()
        models.append(bkg_model)
        dataset.models = models
        
        return dataset
    
    

    def create_dataset_N(self):
        dataset_ = self.create_dataset()
        dataset_N = MapDataset(
                counts=dataset_.counts.copy(),
                exposure=dataset_.exposure.copy(),
                background=dataset_.background.copy(),
                psf=dataset_.psf.copy(),
                edisp=dataset_.edisp.copy(),
                mask_safe=dataset_.mask_safe.copy(),
                gti=dataset_.gti.copy(),
                name='dataset N')
        models = self.set_model()
        #bkg model
        bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        #irf model
        IRFmodel = IRFModel(dataset_name = dataset_N.name)
        IRFmodel.parameters['tilt_nuisance'].frozen  = False
        IRFmodel.parameters['bias'].frozen  = True
        IRFmodel.parameters['resolution'].frozen  = True
        models.append(IRFmodel)
        dataset_N.models = models
        return dataset_N