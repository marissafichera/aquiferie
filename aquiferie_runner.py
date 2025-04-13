
from embeddingsIE import setup_model, do_model
import embeddingsIE



new_studyareas = ['BrackishWater',
                  'AprilReports_Misc']


studyareas = ['SanJuanBasin',
              'RioArribaCounty',
              'AcomaBasin',
              'RioSanJoseBasin',
              'SanAgustinPlains',
              'GilaSanFrancisco',
              'MimbresBasin',
              'BootheelBasinAndRange',
              'RatonBasin',
              'UnionCounty',
              'EstanciaBasin',
              'HighPlains',
              'SacramentoMountainsPecosSlope',
              'RoswellArtesianBasin',
              'DelawareBasin',
              'CapitanReefAquifer',
              'SaltBasin',
              'SanLuisBasin',
              'EspanolaBasin',
              'SantoDomingoBasin',
              'AlbuquerqueBasin',
              'SocorroBasin',
              'LaJenciaBasin',
              'SanMarcialBasin',
              'EngleBasin',
              'PalomasBasin',
              'JornadaDelMuertoBasin',
              'TularosaBasin',
              'MesillaBasin',
              'MISC_AREAS']


def main():

    for s in new_studyareas:
        cfg = setup_model(s)
        do_model(cfg)



if __name__ == '__main__':
    main()