import numpy as np
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    SkyModel,
    PowerLawNuisanceSpectralModel,
    PowerLawNormNuisanceSpectralModel)
#from MapDatasetNuisanceE import MapDatasetNuisanceE
from gammapy.modeling import Parameter, Parameters
from gammapy.datasets import MapDataset

path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'




class sys_dataset():
    def __init__(self, 
                 dataset_asimov,
                 factor,
                 rnd):
        self.dataset_asimov = dataset_asimov
        self.factor = factor
        self.rnd = rnd
        
    def set_model(self):
        models = Models.read(f"{path_crab}/standard_model.yml").copy()
        model_spectrum  = PowerLawSpectralModel(
            index=2.3,
            amplitude="1e-12 TeV-1 cm-2 s-1",    )
        source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                               spectral_model = model_spectrum,
                               name = "Source")    
        source_model.parameters['lon_0'].frozen = True
        source_model.parameters['lat_0'].frozen = True
        models = Models(source_model)
        return models
    
    def create_dataset(self):
        dataset = self.dataset_asimov.copy()
        exposure = dataset.exposure.copy()
        exposure.data *= (1-self.factor)
        background = dataset.background.copy()
        #background.data *= (1-self.factor)
        dataset.exposure = exposure
        dataset.background = background
        if self.rnd:
            counts_data = np.random.poisson(self.dataset_asimov.counts.data)
        else:
            counts_data = self.dataset_asimov.counts.data

        dataset.counts.data = counts_data
        models = self.set_model()
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        dataset.models = models
        return dataset
    
    
    def set_model_N(self):
        models = Models.read(f"{path_crab}/standard_model.yml").copy()
        model_spectrum  = PowerLawNuisanceSpectralModel(
            index=2.3,
            amplitude="1e-12 TeV-1 cm-2 s-1",  
            amplitudeN = 0)
        print(len(model_spectrum.parameters))
        print(len(model_spectrum.default_parameters))

        source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                               spectral_model = model_spectrum,
                               name = "SourceN")  
        source_model.parameters['lon_0'].frozen = True
        source_model.parameters['lat_0'].frozen = True
        models = Models(source_model)
        return models



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
        models = self.set_model_N()
        #bkg_spectralmodel = PowerLawNormNuisanceSpectralModel(
        #            tilt=1,
        #            norm=1,
        #            norm_nuisance=0
        #)
        bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name,
                                      )#spectral_model = bkg_spectralmodel)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        dataset_N.models = models
        return dataset_N