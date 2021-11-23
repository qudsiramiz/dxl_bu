
;BMW 2017, March 17
; prints out cusp position in GSE

;------Input Date and time ----
year = 2015
month = 6
day = 1
hour = 0
minute = 0
second = 0

; cusp position or radial distance from Earth in RE
rho = 8.0 ; altitude in re
;------------------------------

; Dont change these ======================
; Constants from Tsyganenko and Russell JGR 1999
phi_c0 = 0.24
alpha1 = 0.1287
alpha2 = 0.0314
;=====================================


geopack_recalc, year, month,day,hour, minute, second, tilt=tilt, /date ; Get the dipole tilt from the input date
psy = tilt*!PI/180. ; Dipole tilt in radians 
;--- to find southern cusp, flip the sign of the psy variable

; Calculations based on Tsyganenko and Russell model
phi_1 = phi_c0 - (alpha1*psy+alpha2*psy^2)
x1 = sqrt(rho)
x2 = sqrt(rho+sin(phi_1)^(-2)-1)
phi_c = asin(x1/x2)+psy

; Convert to cartesian in SM coordinates
z_sm = rho*sin(!PI/2.-phi_c)
x_sm = rho*cos(!PI/2.-phi_c)
y_sm = 0. ; model assumes cusp is at y=0.

; Convert from SM to another coordinate system
GEOPACK_CONV_COORD, x_sm,y_sm,z_sm,xn_gse, yn_gse, zn_gse, /from_SM, /to_GSE

print, 'Cusp northern position in GSE [RE], X, Y, Z'
print, xn_gse, yn_gse, zn_gse
stop



end