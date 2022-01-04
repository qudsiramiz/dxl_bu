# Import the required modules

import h5py as hf
import numpy as np
import datetime

from glob import glob

# Downloads the MFI data from WIND spacecraft from CDAWeb to the
# local directory. There are two different inputs. 1. The start time of  MFI
# data and 2. The length of duration ( perefrably in seconds ). The start time
# could be entered as just a string for the date in the following format:
# 'YYYYMMDDHHmmSS'

## Note: This code CANNOT function without these three additional codes:

# 1. janus_mfi_arcv_hres.py
# 2. janus_mfi_arcv_lres.py
# 3. janus_time.py

plt.close( 'all' )

###############################################################################
## Function to calculate non-linear omega
###############################################################################

def calc_omnl( dat_mag, dat_bam, i ) :

	print i
	pi = np.pi
	qp = 1.6e-19   # Charge of proton in C
	mp = 1.67e-27  # Mass of proton in kg
	mu0 = 4.0*np.pi*1e-7    # Vacuum permeability 

	# High-cadence Magnetic field

	bxh = b_x*1.e-9
#	bxh = dat_mag['b_x_high'][:]*1e-9
#	t   = dat_mag['time_mag'][:]
	dt   = 0.5*(t[2]-t[0])

	# Btot for the period

	bx = dat_bam['b_x'][i]*1e-9 # nT to T
	by = dat_bam['b_y'][i]*1e-9
	bz = dat_bam['b_z'][i]*1e-9

	btot = np.sqrt(bx**2 + by**2 + bz**2)   # nT to T

	# Density

	ne = dat_bam['n_p'][i]
	ne = ne*1e6    #/cm^3 to /m^3

	vax = bxh/np.sqrt(mu0*mp*ne)   # Alfven unit

	n = len(t)

	# Calculate di in km

	di = 2.28e2/((ne/1e6)**0.5)

	# Calculate Vsw in km/s

	Vsw = -dat_bam['v_x_p'][i] 
	dl = 1.0 	# dl = 1di
	dl = dl*di	# di to km

	dtau = dl/Vsw	# Spatial lag to time lag

	m = int(dtau/dt)
	#print m

	# Go to the mid point of high-cadence B
	# and calculate nlin-time by going +/- m steps

	db = vax[int(n/2)+m] - vax[int(n/2)-m]	# x is the longitudinal dir.
	tnl = (dl*1e3)/abs(db)  # unit s
	omcp = (qp*btot)/mp	# proton cyclotron frequency
	omnl =  (2.0*np.pi)/(tnl*omcp)   # in units of 1/omcp

	return omnl

###############################################################################
## MFI Data download ( 90 seconds hard-coded )
###############################################################################

# Downloads the MFI data from WIND spacecraft from CDAWeb to the
# local directory. There are two different inputs. 1. The start time of  MFI
# data and 2. The length of duration ( perefrably in seconds ). The start time
# could be entered as just a string for the date in the following format:
# 'YYYYMMDDHHmmSS'

## Note: This code CANNOT function without these three additional codes:

# 1. janus_mfi_arcv_hres.py
# 2. janus_mfi_arcv_lres.py
# 3. janus_time.py

def mfi_data_dwnld( time=None, dur=None, res='low' ) :

	# Load the modules necessary for loading Wind/FC and Wind/MFI data.

	from janus_mfi_arcv_hres import mfi_arcv_hres
	from janus_mfi_arcv_lres import mfi_arcv_lres

	# Enter the MFI resolution to be downloaded.

	if( res == 'low' ) :

		arcv = mfi_arcv_lres( )

	elif( res == 'high' ) :

		arcv = mfi_arcv_hres( )

	# Extract the year, month, day etc from the datetime object

	yy = str( time.year   )
	mm = str( time.month  )
	dd = str( time.day    )
	hh = str( time.hour   )
	MM = str( time.minute )
	ss = str( time.second )

	date = yy + '-' + mm + '-' + dd + '-' + hh + '-' + MM + '-' + ss

	try :

		( mfi_t, mfi_b_x, mfi_b_y, mfi_b_z ) = arcv.load_rang( date, dur )

	except:

		pass

	mfi_b_vec = array( [ [ mfi_b_x[i], mfi_b_y[i], mfi_b_z[i] ]
	                           for i in range( len( mfi_t ) ) ] )

	return mfi_t, array( mfi_b_x )

###############################################################################

# Read the data created by Ben

dat_bam =  hf.File( 'data/bam_dvapmulti.hf', 'r+' )

key_list = list( dat.keys() )

# Conver the time from the HDF file to a datetime object
# Note: 't_sec' is the time which Ben used for his work. In that case,
# 1994/11/01/00:00:00 was defined as t = 0. 'time_new' is the time in UNIX
# format.

time_new = array( dat['time_new'] )

# Conver the 'time_new' to a datetime object ( in UTC format ).

time_new_full = array( [ datetime.datetime.utcfromtimestamp( xx ) for xx in time_new ] )

# Define the default UNIX time

t0 = datetime.datetime(1970, 1, 1)

dat_gam = hf.File( 'data/gamma_files/gamma_k_r_b_WIND.h5', 'r+' )

fname_mag = glob( 'data/mag_files/*' )

fname_mag = sort( fname_mag )

omnl = full( shape( fname_mag ), nan )

for ind in range( len( fname_mag ) ) :
#for ind in range( 5 ) :

	dat_mag = hf.File( fname_mag[ind], 'r+' )

	omnl[ind] =  calc_omnl( dat_mag, dat_bam, ind )


fname_omega = 'data/gamma_k_r_b_omnl_WIND_v2.h5'

hf_f = hf.File( fname_omega, 'w' )

#hf_f.create_dataset( 'time', data=dat_bam['time_new'] )
hf_f.create_dataset( 'omnl', data=omnl )

for key in dat_gam.keys( ) :

	hf_f.create_dataset( key, data=dat_gam[key] )

hf_f.close()

dat_omnl = hf.File( 'data/gamma_k_r_b_omnl_WIND_v2.h5', 'r+' )

test