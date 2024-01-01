import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as multi
import os 
import pmagpy
import pmagpy.pmag as pmag
import pmagpy.ipmag as ipmag
import pmagpy.pmagplotlib as pmagplotlib
import re
import scipy.integrate as integrate
import scipy.stats as stats
from scipy.optimize import curve_fit
import seaborn as sns
import SPD.lib.leastsq_jacobian as lib_k
import sys

from datetime import datetime as dt
from importlib import reload
from multiprocessing import Pool

import pmagpy.tsunashawfuncs as ts
import importlib


font1={'family': 'Times New Roman','weight': 'normal','size': 20}
font2={'family': 'Times New Roman','weight': 'normal','size': 18}
font3={'family': 'Times New Roman','weight': 'normal','size': 12}

def process_R_data(input_file, output_file, cuttings):
    """ Generate R data at selected AF cuttings
    Paramters
    _________
        input_file : file name of input data with MagIC format
        output_file : file name of the output data
        cuttings : The selected AF cuttings, with the number of cuttings <= 10
    Returns
    ________

    """
    Raw = pd.read_csv("MagIC/%s"%input_file, sep='\t', header=1)
    Specimen = []

    for cutting in cuttings:
        NRM, TRM1, ARM0, ARM1, R0, R1 = [], [], [], [], [], []
        Count=0
        for i in range(len(Raw.specimen)):
            if i == 0:
                Specimen.append(Raw.specimen[i])
            elif Raw.specimen[i] not in Specimen:
                Specimen.append(Raw.specimen[i])
        
            if 0 < i < len(Raw.specimen) - 1:
                if cutting != 'NRM' and cutting != 'NRM0':
                    if (Raw.description[i][0:3] == 'NRM') and (float(Raw.treat_ac_field[i]) == cutting):
                        X0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Y0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Z0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        Count=1
                    if (Raw.description[i][0:3] == 'NRM') and (Raw.description[i + 1][0:3] != 'NRM'):
                        X1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Y1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Z1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        NRM.append(np.sqrt((X0 - X1) ** 2 + (Y0 - Y1) ** 2 + (Z0 - Z1) ** 2))

                    if (Raw.description[i][0:4] == 'TRM1') and (float(Raw.treat_ac_field[i]) == cutting):
                        TX0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TY0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TZ0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                    if (Raw.description[i][0:4] == 'TRM1') and (Raw.description[i + 1][0:3] != 'TRM'):
                        TX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        TRM1.append(np.sqrt((TX0 - TX1) ** 2 + (TY0 - TY1) ** 2 + (TZ0 - TZ1) ** 2))

                    if (Raw.description[i][0:4] == 'ARM0') and (float(Raw.treat_ac_field[i]) == cutting):
                        AX0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AY0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AZ0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                    if (Raw.description[i][0:4] == 'ARM0') and (Raw.description[i + 1][0:3] != 'ARM'):
                        AX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        ARM0.append(np.sqrt((AX0 - AX1) ** 2 + (AY0 - AY1) ** 2 + (AZ0 - AZ1) ** 2))

                    if (Raw.description[i][0:4] == 'ARM1') and (float(Raw.treat_ac_field[i]) == cutting):
                        aX0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aY0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aZ0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                    if (Raw.description[i][0:4] == 'ARM1') and (Raw.description[i + 1][0:3] != 'ARM'):
                        aX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        ARM1.append(np.sqrt((aX0 - aX1) ** 2 + (aY0 - aY1) ** 2 + (aZ0 - aZ1) ** 2))
                
                elif cutting == 'NRM0':
                    if (Raw.description[i][0:3] == 'NRM') and (Raw.description[i - 1][0:4] == 'NRM0'):
                        X0 = float(Raw['magn_moment'][i-1]) * np.cos(float(Raw['dir_inc'][i-1]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i-1]) * np.pi / 180)
                        Y0 = float(Raw['magn_moment'][i-1]) * np.cos(float(Raw['dir_inc'][i-1]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i-1]) * np.pi / 180)
                        Z0 = float(Raw['magn_moment'][i-1]) * np.sin(float(Raw['dir_inc'][i-1]) * np.pi / 180)
                    if (Raw.description[i][0:3] == 'NRM') and (Raw.description[i + 1][0:3] != 'NRM'):
                        X1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Y1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Z1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        NRM.append(np.sqrt((X0 - X1) ** 2 + (Y0 - Y1) ** 2 + (Z0 - Z1) ** 2))

                    if (Raw.description[i][0:4] == 'TRM1') and (Raw.description[i - 1][0:3] != 'TRM'):
                        TX0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TY0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TZ0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                    if (Raw.description[i][0:4] == 'TRM1') and (Raw.description[i + 1][0:3] != 'TRM'):
                        TX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        TRM1.append(np.sqrt((TX0 - TX1) ** 2 + (TY0 - TY1) ** 2 + (TZ0 - TZ1) ** 2))

                    if (Raw.description[i][0:4] == 'ARM0') and (Raw.description[i - 1][0:3] != 'ARM'):
                        AX0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AY0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AZ0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                    if (Raw.description[i][0:4] == 'ARM0') and (Raw.description[i + 1][0:3] != 'ARM'):
                        AX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        ARM0.append(np.sqrt((AX0 - AX1) ** 2 + (AY0 - AY1) ** 2 + (AZ0 - AZ1) ** 2))

                    if (Raw.description[i][0:4] == 'ARM1') and (Raw.description[i - 1][0:3] != 'ARM'):
                        aX0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aY0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aZ0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                    if (Raw.description[i][0:4] == 'ARM1') and (Raw.description[i + 1][0:3] != 'ARM'):
                        aX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        ARM1.append(np.sqrt((aX0 - aX1) ** 2 + (aY0 - aY1) ** 2 + (aZ0 - aZ1) ** 2))
                        
                elif cutting == 'NRM':
                    if (Raw.description[i][0:3] == 'NRM') and (Raw.description[i - 1][0:4] == 'NRM0'):
                        X0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Y0 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Z0 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                    if (Raw.description[i][0:3] == 'NRM') and (Raw.description[i + 1][0:3] != 'NRM'):
                        X1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Y1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        Z1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        NRM.append(np.sqrt((X0 - X1) ** 2 + (Y0 - Y1) ** 2 + (Z0 - Z1) ** 2))

                    if (Raw.description[i][0:4] == 'TRM1') and (Raw.description[i - 1][0:3] != 'TRM'):
                        TX0 = float(Raw['magn_moment'][i+1]) * float(np.cos(Raw['dir_inc'][i+1]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i+1]) * np.pi / 180)
                        TY0 = float(Raw['magn_moment'][i+1]) * float(np.cos(Raw['dir_inc'][i+1]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i+1]) * np.pi / 180)
                        TZ0 = float(Raw['magn_moment'][i+1]) * np.sin(Raw['dir_inc'][i+1] * np.pi / 180)
                    if (Raw.description[i][0:4] == 'TRM1') and (Raw.description[i + 1][0:3] != 'TRM'):
                        TX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        TZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        TRM1.append(np.sqrt((TX0 - TX1) ** 2 + (TY0 - TY1) ** 2 + (TZ0 - TZ1) ** 2))

                    if (Raw.description[i][0:4] == 'ARM0') and (Raw.description[i - 1][0:3] != 'ARM'):
                        AX0 = float(Raw['magn_moment'][i+1]) * float(np.cos(Raw['dir_inc'][i+1]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i+1]) * np.pi / 180)
                        AY0 = float(Raw['magn_moment'][i+1]) * float(np.cos(Raw['dir_inc'][i+1]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i+1]) * np.pi / 180)
                        AZ0 = float(Raw['magn_moment'][i+1]) * np.sin(Raw['dir_inc'][i+1] * np.pi / 180)
                    if (Raw.description[i][0:4] == 'ARM0') and (Raw.description[i + 1][0:3] != 'ARM'):
                        AX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        AZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        ARM0.append(np.sqrt((AX0 - AX1) ** 2 + (AY0 - AY1) ** 2 + (AZ0 - AZ1) ** 2))

                    if (Raw.description[i][0:4] == 'ARM1') and (Raw.description[i - 1][0:3] != 'ARM'):
                        aX0 = float(Raw['magn_moment'][i+1]) * float(np.cos(Raw['dir_inc'][i+1]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i+1]) * np.pi / 180)
                        aY0 = float(Raw['magn_moment'][i+1]) * float(np.cos(Raw['dir_inc'][i+1]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i+1]) * np.pi / 180)
                        aZ0 = float(Raw['magn_moment'][i+1]) * np.sin(Raw['dir_inc'][i+1] * np.pi / 180)
                    if (Raw.description[i][0:4] == 'ARM1') and (Raw.description[i + 1][0:3] != 'ARM'):
                        aX1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.cos(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aY1 = float(Raw['magn_moment'][i]) * np.cos(float(Raw['dir_inc'][i]) * np.pi / 180) * np.sin(
                            float(Raw['dir_dec'][i]) * np.pi / 180)
                        aZ1 = float(Raw['magn_moment'][i]) * np.sin(float(Raw['dir_inc'][i]) * np.pi / 180)
                        ARM1.append(np.sqrt((aX0 - aX1) ** 2 + (aY0 - aY1) ** 2 + (aZ0 - aZ1) ** 2))                   
        if Count==0:
            print('please select an appropriate AF cutting instead of %f.'%cutting)
            return

        for i in range(len(NRM)):
            R0.append(NRM[i] / ARM0[i])
        for i in range(len(TRM1)):
            R1.append(TRM1[i] / ARM1[i])
        
        if cutting == cuttings[0]:
            R = pd.DataFrame({'Specimen': Specimen, 'R0_base': R0, 'R1_base': R1})
        elif type(cutting) == str:
            R.insert(R.shape[1],'R0_%s'%cutting,R0)
            R.insert(R.shape[1],'R1_%s'%cutting,R1)
        elif type(cutting) == float:
            R.insert(R.shape[1],'R0_%.3f'%cutting,R0)
            R.insert(R.shape[1],'R1_%.3f'%cutting,R1)
    R.to_csv("csv/%s.csv"%output_file, index=False)



def process_PINT_data(MagIC_directory='MagIC/',AF_min=0,AF_max=180):
    """ Calculate paleointensity data from the selected AF range, using the code from (Yamamoto et. al, 2022) with minor modification.
    Paramters
    _________
        input_file : file name of input data with MagIC format
        AF_min : the minimum coercivity
        AF_max : the maximum coercivity
    Returns
    ________

    """
    importlib.reload(ts)

    analysis='ordinary'
    minN=4 # minimum number of data points
    minR=0.0 # minimum correlation coefficient of a slope
    minSlopeT=-10 # minimum slope of TRM1-TRM2* plot
    maxSlopeT=10 # minimum slope of TRM1-TRM2* plot
    minFrac=0.00 # minimum frac of NRM-TRM1* and TRM1-TRM2* plots
    maxKrv=10 # consideration for Lloyd et al.(2021)
    maxBeta=10 # consideration for Lloyd et al.(2021)
    maxFresid=1 # consideration for Lloyd et al.(2021)

    MagIC_directory=MagIC_directory
    Plot_directory='plots/'
    csv_directory='csv/'
    dir_path='Figures/'

    lat,lon=0, 0
    height=''
    location='na'
    location_type='outcrop'

    age="0"
    age_min,age_max="0","0"
    age_unit='Ma'

    classes='intrusive:igneous'
    geologic_types='Intrusives'
    lithologies='basalt'
    citations='This study'

    method_codes_first='LT-NRM-PAR:LP-PI-TRM:IE-SH:LP-LT:DA-ALT-RS:DA-AC-AARM'
    method_codes_second='LT-TRM-PAR:LP-PI-TRM:IE-SH:LP-LT:DA-ALT-RS:DA-AC-AARM'

    ## template files
    sample_df=pd.read_csv(MagIC_directory+'samples.txt',sep='\t',header=1)
    sample_df=sample_df.rename(columns={'sample': 'sample_name'})
    sample_df=sample_df.sort_values('sample_name')

    sids=sample_df.site.unique()
    siteinfo_df=pd.DataFrame(sids)
    siteinfo_df.columns=['site']

    siteinfo_df['age']=age
    siteinfo_df['age_high']=age_max
    siteinfo_df['age_low']=age_min
    siteinfo_df['age_unit']=age_unit

    siteinfo_df['lat']=lat
    siteinfo_df['lon']=lon
    siteinfo_df['height']=height

    siteinfo_df['geologic_classes']=classes
    siteinfo_df['geologic_types']=geologic_types
    siteinfo_df['lithologies']=lithologies
    siteinfo_df['location']=location
    siteinfo_df['location_type']=location_type

    siteinfo_df['citations']=citations


    siteinfo_df.to_csv(csv_directory+'siteinfo_template.csv', index=None)

    spec_df = pd.read_csv(MagIC_directory+'specimens.txt', sep='\t', header=1)
    spec_df = spec_df.sort_values('specimen')
    sids = spec_df.specimen.unique()
    step_df = pd.DataFrame(sids)
    step_df.columns = ['specimen']

    step_df['zijd_min'] = AF_min
    step_df['zijd_max'] = AF_max
    step_df['trm1_star_min'] = AF_min
    step_df['trm1_star_max'] = AF_max
    step_df['trm2_star_min'] = AF_min
    step_df['trm2_star_max'] = AF_max

    step_df.to_csv(csv_directory+'steps.csv', index=None)

    plot_files = os.listdir(Plot_directory)
    for file in plot_files:
        if '.pdf' in file:
            os.remove(Plot_directory+file)

    meas = pd.read_csv(MagIC_directory+'measurements.txt', sep='\t', header=1)
    meas['dir_dec_diff']=""
    meas['dir_inc_diff']=""
    meas['magn_mass_diff']=""
    meas['treat_ac_field_mT']=meas['treat_ac_field']*1e3 # for convenience


    sids=meas.specimen.unique()

    step_wishes=True
    try:
        steps_df=pd.read_csv(csv_directory+'steps.csv')
    except IOError:
        step_wishes=False
        steps_df=pd.DataFrame(sids)
        steps_df.columns=['specimen']
        steps_df=steps_df.set_index('specimen')
        #meas=meas.set_index(['specimen','description'])
        for sid in sids:
            meas_n=meas[ (meas.description.str.contains('NRM')==True)
                     & (meas.specimen.str.contains(sid)==True)]
            meas_t1=meas[ (meas.description.str.contains('TRM1')==True)
                     & (meas.specimen.str.contains(sid)==True)]
            meas_t2=meas[ (meas.description.str.contains('TRM2')==True)
                     & (meas.specimen.str.contains(sid)==True)]
            if(len(meas_n)>0):
                meas_n=meas_n.set_index(['specimen','description'])
                steps_df.loc[sid,'zijd_min']=meas_n.loc[sid,'NRM'].treat_ac_field_mT.min()
                steps_df.loc[sid,'zijd_max']=meas_n.loc[sid,'NRM'].treat_ac_field_mT.max()
            if(len(meas_t1)>0):
                meas_t1=meas_t1.set_index(['specimen','description'])
                steps_df.loc[sid,'trm1_star_min']=meas_t1.loc[sid,'TRM1'].treat_ac_field_mT.min()
                steps_df.loc[sid,'trm1_star_max']=meas_t1.loc[sid,'TRM1'].treat_ac_field_mT.max()
            if(len(meas_t2)>0):
                meas_t2=meas_t2.set_index(['specimen','description'])
                steps_df.loc[sid,'trm2_star_min']=meas_t2.loc[sid,'TRM2'].treat_ac_field_mT.min()
                steps_df.loc[sid,'trm2_star_max']=meas_t2.loc[sid,'TRM2'].treat_ac_field_mT.max()
        steps_df=steps_df.reset_index()
        meas=meas.reset_index()
        
    importlib.reload(ts)
    ## remanence type
    ntrm_types=['NRM','TRM1','TRM2']
    arm_types=['ARM0','ARM1','ARM2']

    ## prepare dataframe for this specimen for private data output
    spec_prv_df=pd.DataFrame(sids)
    spec_prv_df.columns=['specimen']
    spec_prv_df=spec_prv_df.set_index('specimen')

    ## prepare for main loop
    cnt=1 # plot counter
    sidPars=[] # list of dictionaries for specimen intensity statistics
    stepRecs=[] # list of step wishes after optimization 

    ## step through the specimens
    ## for sid in ['SW01-01A-1_sun','SW01-02A-1_sun', 'SW01-03A-1_sun']:
    for sid in sids:
        print ('--------------')
        print ('[working on: ', sid, ']')
    
        ## make a dictionary for this specimen's statistics
        sidpars_nrm={}
        sidpars_nrm['specimen']=sid
        sidpars_nrm['int_corr']='c'
        sidpars_nrm['citations']=citations
        sidpars_nrm['method_codes']=method_codes_first
        #
        sidpars_trm={}
        sidpars_trm['specimen']=sid
        sidpars_trm['int_corr']='c'
        sidpars_trm['citations']=citations
        sidpars_trm['method_codes']=method_codes_second
        #
        steprec={}
        steprec['specimen']=sid
    
        ## select the measurement data for this specimen
        sid_data=meas[meas.specimen==sid]

        ## substitute lab_field
        aftrm10=sid_data[sid_data.description.str.contains('TRM10')]
        if (len(aftrm10)>0): lab_field=aftrm10.treat_dc_field.tolist()[0]

        ## preparation of data
        sid_data,sid_df,afnrm,aftrm1,aftrm2,afarm0,afarm1,afarm2 \
            =ts.prep_sid_df(ntrm_types+arm_types,sid_data)

        ## data output: AF vs XRM intensity
        sid_df.to_csv(csv_directory+sid+'_af_vs_xrms.csv',index=True)

        ## get the treatment steps for this specimen
        #print(steps_df)
        sid_steps=steps_df[steps_df.specimen==sid].head(1)
        #print(sid_steps)
        if (len(afnrm)>0):
            zijd_min,zijd_max=sid_steps.zijd_min.values[0],sid_steps.zijd_max.values[0]
        if (len(aftrm1)>0) & (len(afarm1)>0):
            trm1_star_min,trm1_star_max=sid_steps.trm1_star_min.values[0],sid_steps.trm1_star_max.values[0]
        if (len(aftrm2)>0) & (len(afarm2)>0):
            trm2_star_min,trm2_star_max=sid_steps.trm2_star_min.values[0],sid_steps.trm2_star_max.values[0]
    

        ## Fig1: AF vs XRM intensity
        ##       with output of relevant data 
        fig1=plt.figure(cnt,figsize=(15,22))
        fig1.clf()
        ax1,ax2,ax3,ax4,ax5,ax6 =\
            fig1.add_subplot(321),fig1.add_subplot(322),fig1.add_subplot(323),\
            fig1.add_subplot(324),fig1.add_subplot(325),fig1.add_subplot(326)
        for t in (ntrm_types+arm_types):
            [x1,x2,x3,x4,x5]=[0,0,0,0,0]
            if (t=='NRM')  & (len(afnrm)>0):
                x1,x2,x3,x4,x5=ts.plot_af_xrm(sid,sid_data,ax1,afnrm,t)
            if (t=='TRM1') & (len(aftrm1)>0):
                x1,x2,x3,x4,x5=ts.plot_af_xrm(sid,sid_data,ax3,aftrm1,t)
            if (t=='TRM2') & (len(aftrm2)>0):
                x1,x2,x3,x4,x5=ts.plot_af_xrm(sid,sid_data,ax5,aftrm2,t)
            if (t=='ARM0') & (len(afarm0)>0):
                x1,x2,x3,x4,x5=ts.plot_af_xrm(sid,sid_data,ax2,afarm0,t)
                rem_ltd_perc=100.* (x4 - 1.0)/x4
                spec_prv_df.loc[sid,'ltd%_ARM0']='%7.1f'%(rem_ltd_perc)
                sidpars_nrm['rem_ltd_perc']='%7.1f'%(rem_ltd_perc)
            if (t=='ARM1') & (len(afarm1)>0):
                x1,x2,x3,x4,x5=ts.plot_af_xrm(sid,sid_data,ax4,afarm1,t)
                rem_ltd_perc_1=100.* (x4 - 1.0)/x4
                spec_prv_df.loc[sid,'ltd%_ARM1']='%7.1f'%(rem_ltd_perc_1)
            if (t=='ARM2') & (len(afarm2)>0):
                x1,x2,x3,x4,x5=ts.plot_af_xrm(sid,sid_data,ax6,afarm2,t)
                rem_ltd_perc_2=100.* (x4 - 1.0)/x4
                spec_prv_df.loc[sid,'ltd%_ARM2']='%7.1f'%(rem_ltd_perc_2)
            if (abs(x1+x2+x3+x4+x5)>0):
                afdmax=x1
                s1='MDF_'+str(t)
                s2=str(t)+'_0mT(uAm2/kg)'
                s3=str(t)+'0/'+str(t)+'_0mT'
                s4=str(t)+'_'+str(int(afdmax))+'mT/'+str(t)+'_0mT'
                #print(s1, s2, s3, s4)
                spec_prv_df.loc[sid,s1],spec_prv_df.loc[sid,s2],\
                spec_prv_df.loc[sid,s3],spec_prv_df.loc[sid,s4]\
                    =float('{:.4e}'.format(x2)),float('{:.4e}'.format(x3)),\
                        float('{:.4e}'.format(x4)),float('{:.4e}'.format(x5))
            #print(spec_prv_df)
        fig1.savefig(Plot_directory + sid + '_afdemag.pdf')
        cnt = cnt + 1
        
        ## opt interval search (min MAD, DANG) for Zijderveld
        if (step_wishes == False) & (len(afnrm)>0):
            zijd_max=afnrm['treat_ac_field_mT'].tolist()[len(afnrm)-1]
            zijd_min, zijd_max = ts.opt_interval_zij(afnrm, minN)
            steprec['zijd_min'],steprec['zijd_max'] = zijd_min, zijd_max

        ## Fig2: Zijderveld plot for the NRM demag directions
        ##       with output of relevant data 
        fig2=plt.figure(cnt,figsize=(14,7))
        fig2.clf()
        ax1, ax2 =fig2.add_subplot(121), fig2.add_subplot(122)
        if (len(afnrm)>0):
            x1,x2,x3,x4,x5=ts.plot_zijd(sid, sid_data, ax1, ax2, afnrm, zijd_min, zijd_max)
            sidpars_nrm['dir_dec']='%7.1f'%(x1)
            sidpars_nrm['dir_inc']='%7.1f'%(x2)
            sidpars_nrm['dir_mad_free']='%7.1f'%(x3)
            spec_prv_df.loc[sid,'from_zijd']='%7.1f'%(zijd_min)
            spec_prv_df.loc[sid,'to_zijd']='%7.1f'%(zijd_max)
            spec_prv_df.loc[sid,'PCA_Dec']='%7.1f'%(x1)
            spec_prv_df.loc[sid,'PCA_Inc']='%7.1f'%(x2)
            spec_prv_df.loc[sid,'PCA_MAD']='%7.1f'%(x3)
            spec_prv_df.loc[sid,'PCA_N']='%2d'%(x4)
            spec_prv_df.loc[sid,'PCA_DANG']='%7.1f'%(x5)

        fig2.savefig(Plot_directory + sid + '_zijd.pdf')
        cnt = cnt + 1

        ## Fig3: basic plots (NRM-ARM0, TRM1-ARM1, TRM2-ARM2)
        ##       with output of relevant data 
        fig3=plt.figure(cnt,figsize=(10,10))
        fig3.clf()
        ax1,ax2,ax3 = fig3.add_subplot(221),fig3.add_subplot(222),fig3.add_subplot(223)
        for (t1,t2) in zip(ntrm_types,arm_types):
            [x1,x2,x3,x4,x5]=[0,0,0,0,0]
            if (t1=='NRM') & (t2=='ARM0') & (len(afnrm)>0) & (len(afarm0)>0):
                x1,x2,x3,x4,x5=\
                    ts.plot_ntrm_arm(sid,ax1,sid_df,afnrm,zijd_min,zijd_max,'arm0','nrm')
            if (t1=='TRM1') & (t2=='ARM1') & (len(aftrm1)>0) & (len(afarm1)>0):
                x1,x2,x3,x4,x5=\
                    ts.plot_ntrm_arm(sid,ax2,sid_df,aftrm1,zijd_min,zijd_max,'arm1','trm1')
            if (t1=='TRM2') & (t2=='ARM2') & (len(aftrm2)>0) & (len(afarm2)>0):
                x1,x2,x3,x4,x5=\
                    ts.plot_ntrm_arm(sid,ax3,sid_df,aftrm2,zijd_min,zijd_max,'arm2','trm2')
            if (abs(x1+x2+x3+x4+x5)>0):
                ss=str(t1)+'-'+str(t2)
                [s1,s2,s3,s4,s5]=['slope_'+ss, 'r_'+ss, 'N_'+ss, 'k_'+ss, 'k\'_'+ss, ]
                spec_prv_df.loc[sid,s1]='%7.3f'%(x1)
                spec_prv_df.loc[sid,s2]='%7.3f'%(x2)
                spec_prv_df.loc[sid,s3]='%2d'%(x3)
                spec_prv_df.loc[sid,s4]='%7.3f'%(x4)
                spec_prv_df.loc[sid,s5]='%7.3f'%(x5)

        fig3.savefig(Plot_directory + sid + '_ratios1.pdf')
        cnt = cnt + 1

        ## set the interval for NRM-TRM1* and TRM1-TRM2*
        if (len(afnrm)>0) & (len(aftrm1)>0) & (len(afarm0)>0) & (len(afarm1)>0):
            if (step_wishes == False) & (analysis == 'ordinary'):
                trm1_star_min, trm1_star_max =\
                    ts.opt_interval_first_heating(zijd_min,sid_df,afnrm,minN,minFrac,minR)
            if (analysis == 'best_reg') or (analysis == 'best_krv'):
                combinedRegs1=ts.API_param_combine(sid_df,afnrm,aftrm1,zijd_min,minN) 
                if (analysis == 'best_reg'):
                    trm1_star_min,trm1_star_max,trm2_star_min,trm2_star_max,allrst = \
                        ts.find_best_API_portion_r(combinedRegs1,minFrac,minR,\
                                                   minSlopeT,maxSlopeT)
                if (analysis == 'best_krv'):
                    trm1_star_min,trm1_star_max,trm2_star_min,trm2_star_max,allrst = \
                        ts.find_best_API_portion_k(combinedRegs1,maxBeta,maxFresid,maxKrv,minFrac)
            steprec['trm1_star_min'],steprec['trm1_star_max'] = trm1_star_min, trm1_star_max
        if (len(aftrm1)>0) & (len(aftrm2)>0) & (len(afarm1)>0) & (len(afarm2)>0):
            if (step_wishes == False) & (analysis == 'ordinary'):
                trm2_star_min, trm2_star_max =\
                    ts.opt_interval_second_heating(sid_df,aftrm1,\
                                                   minN,minFrac,minR,minSlopeT,maxSlopeT)
            steprec['trm2_star_min'],steprec['trm2_star_max'] = trm2_star_min, trm2_star_max
    
        ## Fig4: main plots (NRM-TRM1*, TRM1-TRM2*)
        fig4=plt.figure(cnt,figsize=(10,5))
        fig4.clf()
        ax1, ax2 =fig4.add_subplot(121), fig4.add_subplot(122)
        # TRM1 on TRM2*
        [x1,x2,x3,x4,x5,x6,x7,x8]=[0,0,0,0,0,0,0,0]
        if (len(aftrm1)>0) & (len(aftrm2)>0) & (len(afarm1)>0) & (len(afarm2)>0):
            if (analysis == 'ordinary') or (analysis == 'best_reg'):
                x1,x2,x3,x4,x5,x6,x7,x8,x9,ss1 =\
                    ts.plot_pint_main(sid,ax2,sid_df,aftrm1,'trm2_star','trm1',\
                                      trm2_star_min,trm2_star_max,aftrm1,aftrm2,spec_prv_df,\
                                      'reg',minR,minFrac,minSlopeT,maxSlopeT,maxBeta,maxFresid,\
                                      maxKrv,lab_field)
            if (analysis == 'best_krv'):
                x1,x2,x3,x4,x5,x6,x7,x8,x9,ss1 =\
                    ts.plot_pint_main(sid,ax2,sid_df,aftrm1,'trm2_star','trm1',\
                                      trm2_star_min,trm2_star_max,\
                                      aftrm1,aftrm2,spec_prv_df,'krv',\
                                      minR,minFrac,minSlopeT,maxSlopeT,\
                                      maxBeta,maxFresid,maxKrv,lab_field)
        if (abs(x1+x2+x3+x4+x5+x6+x7+x8+x9)>0):
            ss='TRM1-TRM2*'
            [s1,s2]=['from_'+ss, 'to_'+ss]
            spec_prv_df.loc[sid,s1]='%7.1f'%(trm2_star_min)
            spec_prv_df.loc[sid,s2]='%7.1f'%(trm2_star_max)
            [s1,s2,s3,s4,s5,s6,s7,s8,s9] =\
                ['slope_'+ss, 'r_'+ss, 'N_'+ss, 'frac_'+ss, 'dAIC_'+ss, 'k_'+ss, 'k\'_'+ss, 'f_resid_'+ss, 'beta_'+ss]
            spec_prv_df.loc[sid,s1]='%7.3f'%(x1)
            spec_prv_df.loc[sid,s2]='%7.3f'%(x2)
            spec_prv_df.loc[sid,s3]='%2d'%(x3)
            spec_prv_df.loc[sid,s4]='%7.3f'%(x4)
            spec_prv_df.loc[sid,s5]='%7.3f'%(x5)
            spec_prv_df.loc[sid,s6]='%7.3f'%(x6)
            spec_prv_df.loc[sid,s7]='%7.3f'%(x7)
            spec_prv_df.loc[sid,s8]='%7.3f'%(x8)
            spec_prv_df.loc[sid,s9]='%7.3f'%(x9)
            #
            sidpars_trm['description']='Values for the TRM1 normalized by TRM2*'
            sidpars_trm['meas_step_min']=trm2_star_min*1e-3
            sidpars_trm['meas_step_max']=trm2_star_max*1e-3
            #
            sidpars_trm['int_b']='%7.3f'%(x1)
            sidpars_trm['int_abs']=lab_field*x1
            sidpars_trm['int_r2_corr']='%7.3f'%(x2**2)
            sidpars_trm['int_n_measurements']=x3
            sidpars_trm['int_frac']='%7.3f'%(x4)
        # NRM on TRM1*
        [x1,x2,x3,x4,x5,x6,x7,x8]=[0,0,0,0,0,0,0,0]
        if (len(afnrm)>0) & (len(aftrm1)>0) & (len(afarm0)>0) & (len(afarm1)>0):
            if (analysis == 'ordinary') or (analysis == 'best_reg'):
                x1,x2,x3,x4,x5,x6,x7,x8,x9,ss1 =\
                    ts.plot_pint_main(sid,ax1,sid_df,afnrm,'trm1_star','nrm',\
                                      trm1_star_min,trm1_star_max,aftrm1,aftrm2,spec_prv_df,\
                                      'reg',minR,minFrac,minSlopeT,maxSlopeT,maxBeta,maxFresid,\
                                      maxKrv,lab_field)
            if (analysis == 'best_krv'):
                x1,x2,x3,x4,x5,x6,x7,x8,x9,ss1 =\
                    ts.plot_pint_main(sid,ax1,sid_df,afnrm,'trm1_star','nrm',\
                                      trm1_star_min,trm1_star_max,\
                                      aftrm1,aftrm2,spec_prv_df,'krv',\
                                      minR,minFrac,minSlopeT,maxSlopeT,\
                                      maxBeta,maxFresid,maxKrv,lab_field)
        if (abs(x1+x2+x3+x4+x5+x6+x7+x8+x9)>0):
            ss='NRM-TRM1*'
            [s1,s2]=['from_'+ss, 'to_'+ss]
            spec_prv_df.loc[sid,s1]='%7.1f'%(trm1_star_min)
            spec_prv_df.loc[sid,s2]='%7.1f'%(trm1_star_max)
            [s1,s2,s3,s4,s5,s6,s7,s8,s9] =\
                ['slope_'+ss, 'r_'+ss, 'N_'+ss, 'frac_'+ss, 'dAIC_'+ss, 'k_'+ss, 'k\'_'+ss, 'f_resid_'+ss, 'beta_'+ss]
            spec_prv_df.loc[sid,s1]='%7.3f'%(x1)
            spec_prv_df.loc[sid,s2]='%7.3f'%(x2)
            spec_prv_df.loc[sid,s3]='%2d'%(x3)
            spec_prv_df.loc[sid,s4]='%7.3f'%(x4)
            spec_prv_df.loc[sid,s5]='%7.3f'%(x5)
            spec_prv_df.loc[sid,s6]='%7.3f'%(x6)
            spec_prv_df.loc[sid,s7]='%7.3f'%(x7)
            spec_prv_df.loc[sid,s8]='%7.3f'%(x8)
            spec_prv_df.loc[sid,s9]='%7.3f'%(x9)
            if (('rejected' in ss1) == False):
                if (analysis == 'ordinary'):
                    spec_prv_df.loc[sid,'pint(uT)']='%7.3f'%(lab_field*x1*1e6)
                else:
                    spec_prv_df.loc[sid,'min_pint(uT)_passed']='%7.3f'%(lab_field*min(allrst.slope_n)*1e6)
                    spec_prv_df.loc[sid,'max_pint(uT)_passed']='%7.3f'%(lab_field*max(allrst.slope_n)*1e6)
            #
            sidpars_nrm['description']='Values for the NRM normalized by TRM1*'
            sidpars_nrm['meas_step_min']=trm1_star_min*1e-3
            sidpars_nrm['meas_step_max']=trm1_star_max*1e-3
            #
            sidpars_nrm['int_b']='%7.3f'%(x1)
            sidpars_nrm['int_abs']=lab_field*x1
            sidpars_nrm['int_r2_corr']='%7.3f'%(x2**2)
            sidpars_nrm['int_n_measurements']=x3
            sidpars_nrm['int_frac']='%7.3f'%(x4)

        fig4.savefig(Plot_directory + sid + '_corrected_ordinary.pdf')
        cnt = cnt + 1

        ## Fig5: other basic plots (NRM-TRM1, ARM0-ARM1, TRM1-TRM2, ARM1-ARM2)
        fig5=plt.figure(cnt,figsize=(10,10))
        fig5.clf()
        ax1,ax2,ax3,ax4=\
            fig5.add_subplot(221),fig5.add_subplot(222),fig5.add_subplot(223),fig5.add_subplot(224)
        for t in (ntrm_types+arm_types):
            [x1,x2,x3,x4,x5]=[0,0,0,0,0]
            if (t=='NRM') & (len(aftrm1)>0) & (len(afnrm)>0):
                max_xrm=max(sid_df.nrm.max(), sid_df.trm1.max())
                x1,x2,x3,x4,x5=ts.plot_xrm_xrm2_r2(sid,ax1,sid_df,afnrm,'trm1','nrm',\
                                                trm1_star_min,trm1_star_max)
                ss='NRM-TRM1'
                #[s1,s4,s5]=['slope_NRM-TRM1','k_NRM-TRM1','k\'_NRM-TRM1']
            if (t=='TRM1') & (len(aftrm2)>0) & (len(aftrm1)>0):
                max_xrm=max(sid_df.trm1.max(), sid_df.trm2.max())
                x1,x2,x3,x4,x5=ts.plot_xrm_xrm2_r2(sid,ax3,sid_df,aftrm1,'trm2','trm1',\
                                                trm2_star_min,trm2_star_max)
                ss='TRM1-TRM2'
                #[s1,s4,s5]=['slope_TRM1-TRM2','k_TRM1-TRM2','k\'_TRM1-TRM2']
            if (t=='ARM0') & (len(afarm1)>0) & (len(afarm0)>0):
                max_xrm=max(sid_df.arm0.max(), sid_df.arm1.max())
                x1,x2,x3,x4,x5=ts.plot_xrm_xrm2_r2(sid,ax2,sid_df,afnrm,'arm1','arm0',\
                                                trm1_star_min,trm1_star_max)
                ss='ARM0-ARM1'
                #[s1,s4,s5]=['slope_ARM0-ARM1','k_ARM0-ARM1','k\'_ARM0-ARM1']
                slope_a1=x1
                sidpars_nrm['int_corr_arm']='%7.3f'%(slope_a1)
            if (t=='ARM1') & (len(afarm2)>0) & (len(afarm1)>0):
                max_xrm=max(sid_df.arm1.max(), sid_df.arm2.max())
                x1,x2,x3,x4,x5=ts.plot_xrm_xrm2_r2(sid,ax4,sid_df,aftrm1,'arm2','arm1',\
                                                trm2_star_min,trm2_star_max)
                ss='ARM1-ARM2'
                #[s1,s4,s5]=['slope_ARM1-ARM2','k_ARM1-ARM2','k\'_ARM1-ARM2']
                slope_a2=x1
                sidpars_trm['int_corr_arm']='%7.3f'%(slope_a2)
            if [(t=='NRM')|(t=='TRM1')|(t=='ARM0')|(t=='ARM1')]:
                if (abs(x1+x2+x3+x4+x5)>0):
                    [s1,s2,s4,s5]=['slope_'+ss, 'r_'+ss, 'k_'+ss,'k\'_'+ss]
                    spec_prv_df.loc[sid,s1]=float('{:.4e}'.format(x1))
                    spec_prv_df.loc[sid,s2]=float('{:.4e}'.format(x2))
                    spec_prv_df.loc[sid,s4]=float('{:.4e}'.format(x4))
                    spec_prv_df.loc[sid,s5]=float('{:.4e}'.format(x5))
        if(analysis == 'ordinary'):
            fig5.savefig(Plot_directory+sid+'_ratios2_ordinary.pdf')

        cnt=cnt+1
    
        ## data outtput for MagIC
        sidPars.append(sidpars_nrm)
        sidPars.append(sidpars_trm)

        ## record the step intervals
        if (step_wishes == False): stepRecs.append(steprec)

    ## concluding the analysis 
    # data outtput for MagIC
    if(analysis == 'ordinary'):
        pmag.magic_write(MagIC_directory+'specimens_ts_ordinary.txt',sidPars,'specimens')
    # data outtput for private
    if(analysis == 'ordinary'):
        spec_prv_df.to_csv(csv_directory+'specimen_results_ordinary.csv',index=True)
    # data outtput for step intervals
    if (step_wishes == False): 
        pd.DataFrame(stepRecs).to_csv(csv_directory+'steps_optimum.csv',index=None)



def Select_PINT_data(input_file_1='R_Cuttings', input_file_2='specimen_results_ordinary', output_file='PINT_Sel', maxKrv=0.2, maxBeta=0.1, maxFresid=0.1):
    """ Select paleointensity data by the |k| criteria (Lloyd et. al, 2021).
    Paramters
    _________
        input_file_1 : file name of R data from various AF cuttings
        input_file_2 : file name of paleointensity data from the selected coercivity range
        output_file : file name of the selected paleointensities
        maxKrv : the curvature threshold for the |k| criteria
        maxBeta : the β threshold for the |k| criteria
        maxFresid : the residual fraction threshold for the |k| criteria
    Returns
    ________

    """
    RAll=pd.read_csv("csv/%s.csv"%input_file_1)
    PAll=pd.read_csv("csv/%s.csv"%input_file_2)
    maxKrv=maxKrv
    maxBeta=maxBeta
    maxFresid=maxFresid
    PINT=[]
    for i in range(len(RAll["Specimen"])):
        for j in range(len(PAll["specimen"])):
            if PAll["specimen"][j]==RAll["Specimen"][i]:
                if PAll["f_resid_NRM-TRM1*"][j]<=maxFresid and PAll["k'_NRM-TRM1*"][j]<=maxKrv and PAll["beta_NRM-TRM1*"][j]<=maxBeta:
                    PINT.append(PAll["pint(uT)"][j])
                else:
                    PINT.append(0)
    Template=pd.DataFrame({'Specimen' : RAll["Specimen"], 'PINT' : PINT})
    Template.to_csv("csv/%s.csv"%output_file, index=None)




def LoRA_PRR(R_PRR, P_PRR, K_PRR, Reg_PRR, L_PRR):
    """ Calculate LoRA-Shaw results by the "PINT+R-R" method
    Paramters
    _________
        R_PRR : R data from various AF cuttings
        P_PRR : selected paleointensity data
        K_PRR : curvatures of different regressions between selected paleointensities and R0/R1s of various AF cuttings
        Reg_PRR : linear regressions between selected paleointensities and R0/R1s of various AF cuttings
        L_PRR : the peak AC field of various AF cuttings
    Returns
    ________

    """
    R_PRR=R_PRR
    P_PRR=P_PRR
    K_PRR=K_PRR
    Reg_PRR=Reg_PRR
    Labels=L_PRR
    fig=plt.figure(figsize=(7,7))
    ax=plt.gca()
    Min=[];Max=[]
    for i in range(len(R_PRR)):
        Min.append(np.min(R_PRR[i]))
        Max.append(np.max(R_PRR[i]))
    Min_All=np.min(Min)*0.9
    Max_All=np.max(Max)*1.1
    for i in range(len(R_PRR)):
        x=np.linspace(Min_All,Max_All,100)
        y=Reg_PRR[i][1]+Reg_PRR[i][0]*x
        plt.scatter(R_PRR[i], P_PRR, color=np.array(plt.cm.tab10(i)), s=30)
        plt.plot(x, y, linestyle='dashed', color=np.array(plt.cm.tab10(i)), markersize=15, label=Labels[i])
    plt.xlabel('R$_0$/R$_1$',font2)
    plt.ylabel('Paleointensity ($\mu$T)',font2)
    tick_div=5
    tick_x=[float('{:.2f}'.format(Min_All+(Max_All-Min_All)*(i+1)/tick_div)) for i in range(tick_div)]
    tick_y=[float('{:.0f}'.format(np.min(P_PRR)*0.9+(np.max(P_PRR)*1.1-np.min(P_PRR)*0.9)*(i+1)/tick_div)) for i in range(tick_div)]
    
    Reg_Sel=[Reg_PRR[0]];Label_Sel=[Labels[0]]
    for i in range(len(K_PRR)):
        if K_PRR[i]<0.2:
            Reg_Sel.append(Reg_PRR[i+1])
            Label_Sel.append(Labels[i+1])
    PINTC=[];Label_Pair=[]
    Label_Count=0
    for i in range((len(Reg_Sel)-1)):
        for j in range(i+1,(len(Reg_Sel))): 
            if max(Reg_Sel[j][4],Reg_Sel[i][4])<abs(Reg_Sel[j][0]-Reg_Sel[i][0]):
                PINTC.append((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1])
                if Label_Count==0:
                    plt.scatter((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0]), (Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1], linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100, label='LoRA-Shaw (PRR)')
                else:
                    plt.scatter((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0]), (Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1], linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100)
                Label_Pair.append([Label_Sel[i],Label_Sel[j]])
                Label_Count=1
    ax.set_xlim(Min_All,Max_All)
    ax.set_ylim(np.min(P_PRR)*0.9,(np.max(P_PRR)*1.1))
    ax.set_xticks(tick_x)
    ax.set_yticks(tick_y)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='upper left',prop=font3)
    plt.title('LoRA-Shaw (PINT+R-R)' + '\n' + 'PINT = ' + '%.1f'%np.mean(PINTC) + ' ± ' + '%.1f'%np.std(PINTC) + ' ($\mu$T)',font1, pad=10)
    LoRA_PRR_PINT=pd.DataFrame({'AF cuttings' : Label_Pair, 'PINT' : PINTC})
    LoRA_PRR_PINT.to_csv('csv/LoRA_Shaw_PRR.csv', index=None)
    plt.savefig("Figures/LoRA_Shaw_PRR.pdf", dpi=600)



def LoRA_RR(R_RR, P_RR, K_RR, Reg_RR, Reg_RR_P, L_RR):
    """ Calculate LoRA-Shaw results by the "R-R" method
    Paramters
    _________
        R_RR : R data from various AF cuttings
        P_RR : selected paleointensity data
        K_RR : curvatures of different regressions between R0/R1s of various AF cuttings and R0/R1 (baseline)
        Reg_RR : linear regressions between R0/R1s of various AF cuttings and R0/R1 (baseline)
        Reg_RR_P : the linear regression between selected paleointensities and R0/R1s of various AF cuttings 
        L_RR : the peak AC field of various AF cuttings
    Returns
    ________

    """
    R_RR=R_RR
    P_RR=P_RR
    K_RR=K_RR
    Reg_RR=Reg_RR
    Reg_RR_P=Reg_RR_P
    Labels=L_RR
    fig=plt.figure(figsize=(15,7))
    plt.subplots_adjust(wspace=1/7)
    Min=[];Max=[]
    for i in range(1,len(R_RR)):
        Min.append(np.min(R_RR[i]))
        Max.append(np.max(R_RR[i]))
    Min_All=np.min(Min)*0.9
    Max_All=np.max(Max)*1.1
    plt.subplot(1,2,1)
    ax=plt.gca()
    for i in range(len(Reg_RR)):
        x=np.linspace(np.min(R_RR[0])*0.9,np.max(R_RR[0])*1.1,100)
        y=Reg_RR[i][1]+Reg_RR[i][0]*x
        plt.scatter(R_RR[0], R_RR[i+1], color=np.array(plt.cm.tab10(i)), s=30)
        plt.plot(x, y, linestyle='dashed', color=np.array(plt.cm.tab10(i)), markersize=15, label=Labels[i+1])
    plt.xlabel('R$_0$/R$_1$ (baseline)',font2)
    plt.ylabel('R$_0$/R$_1$ (X mT)',font2)
    tick_div=5
    tick_x=[float('{:.2f}'.format(np.min(R_RR[0])*0.9+(np.max(R_RR[0])*1.1-np.min(R_RR[0])*0.9)*(i+1)/tick_div)) for i in range(tick_div)]
    tick_y=[float('{:.2f}'.format(Min_All+(Max_All-Min_All)*(i+1)/tick_div)) for i in range(tick_div)]
    Reg_Sel=[];Label_Sel=[]
    for i in range(len(K_RR)):
        if K_RR[i]<0.2:
            Reg_Sel.append(Reg_RR[i])
            Label_Sel.append(Labels[i+1])
    PINTC_R=[];Label_Pair=[];PINTC=[]
    Label_Count=0
    for i in range((len(Reg_Sel)-1)):
        for j in range(i+1,(len(Reg_Sel))): 
            if max(Reg_Sel[j][4],Reg_Sel[i][4])<abs(Reg_Sel[j][0]-Reg_Sel[i][0]):
                PINTC_R.append((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0]))
                if Label_Count==0:
                    plt.scatter((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0]), (Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1], linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100, label='LoRA-Shaw (RR)')
                else:
                    plt.scatter((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0]), (Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1], linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100)
                Label_Pair.append([Label_Sel[i],Label_Sel[j]])
                Label_Count=1
    ax.set_xlim(np.min(R_RR[0])*0.9,(np.max(R_RR[0])*1.1))
    ax.set_ylim(Min_All,Max_All)
    ax.set_xticks(tick_x)
    ax.set_yticks(tick_y)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='upper left',prop=font3)
    
    plt.subplot(1,2,2)
    ax2=plt.gca()
    for i in range(len(PINTC_R)):
        PINTC.append(Reg_RR_P[0][0]*PINTC_R[i]+Reg_RR_P[0][1])
    x=np.linspace(np.min(R_RR[0])*0.9,np.max(R_RR[0])*1.1,100)
    y=Reg_RR_P[0][1]+Reg_RR_P[0][0]*x
    if len(PINTC)>0:
        plt.scatter(PINTC_R, PINTC, linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100, label='LoRA-Shaw (RR)')
    plt.plot(x, y, linestyle='solid', color='black', markersize=15, label='Baseline regression')
    plt.xlabel('R$_0$/R$_1$ (baseline)',font2)
    plt.ylabel('Paleointensity ($\mu$T)',font2)
    tick_div=5
    tick_x=[float('{:.2f}'.format(np.min(R_RR[0])*0.9+(np.max(R_RR[0])*1.1-np.min(R_RR[0])*0.9)*(i+1)/tick_div)) for i in range(tick_div)]
    tick_y=[float('{:.0f}'.format(np.min(P_RR)*0.9+(np.max(P_RR)*1.1-np.min(P_RR)*0.9)*(i+1)/tick_div)) for i in range(tick_div)]
    ax2.set_xlim(np.min(R_RR[0])*0.9,(np.max(R_RR[0])*1.1))
    ax2.set_ylim(np.min(P_RR)*0.9,np.max(P_RR)*1.1)
    ax2.set_xticks(tick_x)
    ax2.set_yticks(tick_y)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='upper left',prop=font3)
    plt.title('LoRA-Shaw (R-R)' + '\n' + 'PINT = ' + '%.1f'%np.mean(PINTC) + ' ± ' + '%.1f'%np.std(PINTC) + ' ($\mu$T)',font1, pad=10)
    LoRA_RR_PINT=pd.DataFrame({'AF cuttings' : Label_Pair, 'PINT' : PINTC})
    LoRA_RR_PINT.to_csv('csv/LoRA_Shaw_RR.csv', index=None)
    plt.savefig("Figures/LoRA_Shaw_RR.pdf", dpi=600)



def LoRA_PR(R_PR, P_PR, K_PR, Reg_PR, L_PR):
    """ Calculate LoRA-Shaw results by the "PINT-R" method
    Paramters
    _________
        R_PR : R data from various AF cuttings
        P_PR : selected paleointensity data
        K_PR : curvatures of different regressions between selected paleointensities and R0/R1s of various AF cuttings 
        Reg_PR : linear regressions between selected paleointensities and R0/R1s of various AF cuttings
        L_PR : the peak AC field of various AF cuttings
    Returns
    ________

    """
    R_PR=R_PR
    P_PR=P_PR
    K_PR=K_PR
    Reg_PR=Reg_PR
    Labels=L_PR
    fig=plt.figure(figsize=(7,7))
    ax=plt.gca()
    Min=[];Max=[]
    for i in range(len(R_PR)):
        Min.append(np.min(R_PR[i]))
        Max.append(np.max(R_PR[i]))
    Min_All=np.min(Min)*0.9
    Max_All=np.max(Max)*1.1
    for i in range(len(R_PR)):
        x=np.linspace(Min_All,Max_All,100)
        y=Reg_PR[i][1]+Reg_PR[i][0]*x
        plt.scatter(R_PR[i], P_PR, color=np.array(plt.cm.tab10(i)), s=30)
        plt.plot(x, y, linestyle='dashed', color=np.array(plt.cm.tab10(i)), markersize=15, label=Labels[i])
    plt.xlabel('R$_0$/R$_1$',font2)
    plt.ylabel('Paleointensity ($\mu$T)',font2)
    tick_div=5
    tick_x=[float('{:.2f}'.format(Min_All+(Max_All-Min_All)*(i+1)/tick_div)) for i in range(tick_div)]
    tick_y=[float('{:.0f}'.format(np.min(P_PR)*0.9+(np.max(P_PR)*1.1-np.min(P_PR)*0.9)*(i+1)/tick_div)) for i in range(tick_div)]    
    Reg_Sel=[];Label_Sel=[]
    for i in range(len(K_PR)):
        if K_PR[i]<0.2:
            Reg_Sel.append(Reg_PR[i])
            Label_Sel.append(Labels[i])
    PINTC=[];Label_Pair=[]
    Label_Count=0
    for i in range((len(Reg_Sel)-1)):
        for j in range(i+1,(len(Reg_Sel))): 
            if max(Reg_Sel[j][4],Reg_Sel[i][4])<abs(Reg_Sel[j][0]-Reg_Sel[i][0]):                
                PINTC.append((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1])
                if Label_Count==0:
                    plt.scatter((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0]), (Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1], linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100, label='LoRA-Shaw (PR)')
                else:
                    plt.scatter((Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0]), (Reg_Sel[j][1]-Reg_Sel[i][1])/(Reg_Sel[i][0]-Reg_Sel[j][0])*Reg_Sel[i][0]+Reg_Sel[i][1], linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100)
                    Label_Count+=1
                Label_Pair.append([Label_Sel[i],Label_Sel[j]])
    ax.set_xlim(Min_All,Max_All)
    ax.set_ylim(np.min(P_PR)*0.9,(np.max(P_PR)*1.1))
    ax.set_xticks(tick_x)
    ax.set_yticks(tick_y)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='upper left',prop=font3)
    plt.title('LoRA-Shaw (PINT-R)' + '\n' + 'PINT = ' + '%.1f'%np.mean(PINTC) + ' ± ' + '%.1f'%np.std(PINTC) + ' ($\mu$T)',font1, pad=10)
    LoRA_PR_PINT=pd.DataFrame({'AF cuttings' : Label_Pair, 'PINT' : PINTC})
    LoRA_PR_PINT.to_csv('csv/LoRA_Shaw_PR.csv', index=None)
    plt.savefig("Figures/LoRA_Shaw_PR.pdf", dpi=600)



def LoRA_curve_fits(R_CF, P_CF, K_CF, Reg_CF, L_CF):
    """ Calculate LoRA-Shaw curve fit results
    Paramters
    _________
        R_CF : R data from various AF cuttings
        P_CF : selected paleointensity data
        K_CF : curvatures of different regressions between selected paleointensities and R0/R1s of various AF cuttings
        Reg_CF : linear regressions between selected paleointensities and R0/R1s of various AF cuttings
        L_CF : the peak AC field of various AF cuttings
    Returns
    ________

    """
    R_CF=R_CF
    P_CF=P_CF
    K_CF=K_CF
    Reg_CF=Reg_CF
    Labels=L_CF
    fig=plt.figure(figsize=(15,7))
    plt.subplots_adjust(wspace=1/7)
    plt.subplot(1,2,1)
    ax=plt.gca()
    DelR=[[] for i in range(len(Reg_CF)-1)]
    for i in range(1,len(R_CF)):
        for j in range(len(R_CF[0])):
            DelR[i-1].append(R_CF[i][j]-R_CF[0][j])
    for i in range(len(DelR)):
        plt.scatter(R_CF[0], DelR[i], label=Labels[i+1])
    def f_1(x,A,B,C):
        return A*x+B/x+C
    PINT_RCF=[];PINT_CF=[];CF_Cuttings=[]
    for i in range(len(DelR)):
        A,B,C=curve_fit(f_1,R_CF[0],DelR[i])[0]
        x=np.linspace(np.min(R_CF[0])*0.9,np.max(R_CF[0])*1.1,10000)
        y=A*x+B/x+C
        plt.plot(x, y, zorder=1)
        for j in range(len(x)):
            if A*B>0 and A>0:
                if y[j]==np.min(y) and j!=0 and j!=np.max(x):
                    plt.scatter(x[j], y[j], color='black' ,marker="v", s=50, zorder=2)
                    CF_Cuttings.append(Labels[i+1])
                    PINT_RCF.append(x[j])
                    PINT_CF.append(x[j]*Reg_CF[0][0]+Reg_CF[0][1])
            if A*B>0 and A<0:
                if y[j]==np.max(y) and j!=0 and j!=np.max(x):
                    plt.scatter(x[j], y[j], color='black', marker="v", s=50, zorder=2)
                    CF_Cuttings.append(Labels[i+1])
                    PINT_RCF.append(x[j])
                    PINT_CF.append(x[j]*Reg_CF[0][0]+Reg_CF[0][1])
    Min=[];Max=[]
    for i in range(len(DelR)):
        Min.append(np.min(DelR[i]))
        Max.append(np.max(DelR[i]))
    if np.min(Min)>0:
        Min_All=np.min(Min)*0.1
    else:
        Min_All=np.min(Min)*1.5
    if np.max(Max)>0:
        Max_All=np.max(Max)*1.5
    else:
        Max_All=np.max(Max)*0.1
    plt.xlabel('R$_0$/R$_1$ (baseline)',font2)
    plt.ylabel('$\Delta$R$_0$/R$_1$ (X-baseline)',font2)
    tick_div=5
    tick_x=[float('{:.2f}'.format(np.min(R_CF[0])*0.9+(np.max(R_CF[0])*1.1-np.min(R_CF[0])*0.9)*(i+1)/tick_div)) for i in range(tick_div)]
    tick_y=[float('{:.2f}'.format(Min_All+(Max_All-Min_All)*(i+1)/tick_div)) for i in range(tick_div)]
    ax.set_xlim(np.min(R_CF[0])*0.9,(np.max(R_CF[0])*1.1))
    ax.set_ylim(Min_All,Max_All)
    ax.set_xticks(tick_x)
    ax.set_yticks(tick_y)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='upper left',prop=font3)
    
    plt.subplot(1,2,2)
    ax2=plt.gca()
    x=np.linspace(np.min(R_CF[0])*0.9,np.max(R_CF[0])*1.1,10000)
    y=Reg_CF[0][1]+Reg_CF[0][0]*x
    if len(PINT_CF)>0:
        plt.scatter(PINT_RCF, PINT_CF, linewidths=0.75, edgecolors='black', c='gold', marker='*', s=100, label='LoRA-Shaw (Curve fits)', zorder=2)
    plt.plot(x, y, linestyle='solid', color='black', markersize=15, label='Baseline regression', zorder=1)
    plt.xlabel('R$_0$/R$_1$ (baseline)',font2)
    plt.ylabel('Paleointensity ($\mu$T)',font2)
    tick_div=5
    tick_x=[float('{:.2f}'.format(np.min(R_CF[0])*0.9+(np.max(R_CF[0])*1.1-np.min(R_CF[0])*0.9)*(i+1)/tick_div)) for i in range(tick_div)]
    tick_y=[float('{:.0f}'.format(np.min(P_CF)*0.9+(np.max(P_CF)*1.1-np.min(P_CF)*0.9)*(i+1)/tick_div)) for i in range(tick_div)]
    ax2.set_xlim(np.min(R_CF[0])*0.9,(np.max(R_CF[0])*1.1))
    ax2.set_ylim(np.min(P_CF)*0.9,np.max(P_CF)*1.1)
    ax2.set_xticks(tick_x)
    ax2.set_yticks(tick_y)
    plt.tick_params(labelsize=18)
    plt.tick_params(labelsize=18)
    labels = ax2.get_xticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(loc='upper left',prop=font3)
    plt.title('LoRA-Shaw (Curve fits)' + '\n' + 'PINT = ' + '%.1f'%np.mean(PINT_CF) + ' ± ' + '%.1f'%np.std(PINT_CF) + ' ($\mu$T)',font1, pad=10)
    LoRA_CF_PINT=pd.DataFrame({'AF cuttings' : CF_Cuttings, 'PINT' : PINT_CF})
    LoRA_CF_PINT.to_csv('csv/LoRA_Shaw_CF.csv', index=None)
    plt.savefig("Figures/LoRA_Shaw_CF.pdf", dpi=600)



def LoRA_Shaw(input_file_1='R_Cuttings', input_file_2='PINT_Sel', method='PRR', curve_fits=False):
    """ Calculate LoRA-Shaw results
    Paramters
    _________
        input_file_1 : R data from various AF cuttings
        input_file_2 : selected paleointensity data
        method : which calculation method we use ('PRR' for "PINT+R-R", 'RR' for "R-R", 'PR' for "PINT-R")
        curve_fits : whether do the curve fits or not
    Returns
    ________

    """
    input_file_1=input_file_1; input_file_2=input_file_2; method=method; curve_fits=curve_fits
    R=pd.read_csv("csv/%s.csv"%input_file_1)
    PINT=pd.read_csv("csv/%s.csv"%input_file_2)
    Sid=[];SelP=[];Sel=[]
    for i in range(len(PINT['PINT'])):
        if PINT.PINT[i]>0:
            Sid.append(PINT.Specimen[i]);SelP.append(PINT.PINT[i]);Sel.append(i)
    Mean_O=np.mean(SelP);Std_O=np.std(SelP)    
    R_len=int((len(R.columns.values)-1)/2)
    if R_len>10:
        print("Choose too much AF cuttings, please reduce the number of it.")
    else:
        SelR=[[] for i in range(R_len)];Reg_P=[];Reg_R=[];Krv_R=[];Krv_P=[]
        Labels=[]
        count=0
        for i in range(1,R_len*2,2):
            for j in range(len(SelP)):
                SelR[count].append(float(R[R.columns.values[i]][Sel[j]]/R[R.columns.values[i+1]][Sel[j]]))                
            Reg_P.append(stats.linregress(SelR[count],SelP))
            if count>0:
                Reg_R.append(stats.linregress(SelR[0],SelR[count]))
            if i==1:
                Labels.append('Baseline')
            elif R.columns.values[i][3:]!='NRM':
                Labels.append(str(int(float(R.columns.values[i][3:])*1000))+' mT')
            else:
                Labels.append(R.columns.values[i][3:])
            count+=1
        for i in range(len(SelR)-1):
            Krv_R.append(lib_k.AraiCurvature(x=SelR[0],y=SelR[i+1])[0])
        for i in range(len(Krv_P)-1):
            Krv_P.append(lib_k.AraiCurvature(x=SelR[i],y=SelP)[0])
        if method=='PRR':
            LoRA_PRR(SelR, SelP, Krv_R, Reg_P, Labels)
        if method=='RR':
            LoRA_RR(SelR, SelP, Krv_R, Reg_R, Reg_P, Labels)
        if method=='PR':
            LoRA_PR(SelR, SelP, Krv_P, Reg_P, Labels)
        if curve_fits:
            LoRA_curve_fits(SelR, SelP, Krv_R, Reg_P, Labels)
            