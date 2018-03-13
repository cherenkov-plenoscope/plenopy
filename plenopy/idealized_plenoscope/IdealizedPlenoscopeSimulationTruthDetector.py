class IdealizedPlenoscopeSimulationTruthDetector(object):
    def __init__(self, air_shower_photon_ids):
        self.air_shower = air_shower_photon_ids

    def __repr__(self):
        out = self.__class__.__name__
        out += '('
        out += str(self.air_shower.shape[0])+'lixels'
        out += ')'
        return out
