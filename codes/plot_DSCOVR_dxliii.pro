; BMW Dec 2017
; Plot ACE data and make Tsyganenko plot of cusp position
  thm_init
  time_stamp, /off

!Y.STYLE = 1
!X.STYLE = 1
limit = 0
while (limit lt 1) do begin
  ;1. read in file  
  file = '/Users/bmwalsh/Documents/Research/DXL/DXLIII/upload_files/plasma-2-hour.json'
  result  = json_parse(file,/toarray)
  time = time_double(result[1:*,0])
  n = float(result[1:*,1])
  speed = float(result[1:*,2])
  temp = float(result[1:*,3])
      
  flux = speed*n*1e-3
  
  store_data,'flux',dat={x:time, y:flux}
  indx = where((flux lt 100) and (flux gt 0),ct)
  maxv = round(max(flux[indx])+2.)
  if ct lt 2 then maxv=10 ; if statement if there is no data
  
  options, 'flux', yrange = [0, maxv], ytitle = 'Flux', ysubtitle = '10!U8!N [1/cm!U-2!N*s]', $
    psym = symcat(16), symsize = 0.5, panel_size=1.25,constant = 0
  
  indx = where((n lt 500) and (n gt 0.01),ct)
  maxv = round(max(n[indx])+10.)
  if ct lt 2 then maxv=20 ; if statement if there is no data
  store_data,'density',dat={x:time, y:n}
  options, 'density', /ylog, colors = ['r'], ytitle = 'n', ysubtitle = '[cm!U-3!N]', $
    psym = symcat(16), symsize = 0.5, yrange = [0.5,maxv]

  indx = where((speed gt 100) and (flux lt 990),ct)
  maxv = (round(max(speed[indx])/100.)+1.0)*100.
  minv = (round(min(speed[indx])/100.)-1.0)*100.
 if ct lt 2 then begin 
  minv=-500 ; if statement if there is no data
  maxv=-100 ; if statement if there is no data
 endif
   
  store_data,'speed',dat={x:time, y:speed}
  options, 'speed', yrange = [minv,maxv], colors = ['b'], ytitle = 'Bulk Speed', $
    ysubtitle = '[km/s]', psym = symcat(16), symsize = 0.5
  store_data,'temp',dat={x:time, y:temp}
    
  ; read in mag data
  file = '/Users/bmwalsh/Documents/Research/DXL/DXLIII/upload_files/mag-2-hour.json'
  result  = json_parse(file,/toarray)
  time = time_double(result[1:*,0])
  bx = float(result[1:*,1])
  by = float(result[1:*,2])
  bz = float(result[1:*,3])  
    
  store_data,'Bx',dat={x:time, y:bx}
  store_data,'By',dat={x:time, y:by}
  store_data,'Bz',dat={x:time, y:bz}
  
  store_data,'B',data=['Bx','By','Bz']
  maxv = 13
  minv = -13
  options, 'B',constant=0,colors=['b','g','r'],labels=['Bx','By','Bz'],$
    labflag=1,yrange = [minv,maxv], ytitle = 'B GSM', ysubtitle = '[nT]', $
    psym = symcat(16), symsize = 0.4
  
  timespan, time_string(min(time)),2,/hours
  
  tplot_options,'charsize',1.1
  tplot_options,'thick',6.0
  tplot_options,'title','2 Hours DSCOVR, Real-Time'

  ;tplot_options,'font',2.0
  popen, '/Users/bmwalsh/Documents/Research/DXL/DXLIII/upload_files/SW_DSCOVR',encapsulated=1,xsize = 6, ysize = 7
  tplot, ['B','density','speed','flux']
  timebar,max(time)-35.*60.
  xyouts, 100,100,'Plot Generated:'+systime(), /device
  pclose

  print, 'Complete making solar wind image'
;  stop
  ;====================================================
  ;3 Make cusp image
  ; Get values at earth assuming 35 min delay
  indx = closest(time,max(time)-35.*60.)
  parmod=fltarr(10)
  parmod[*]=0.
  parmod[0]=(speed[indx]*1000.)^2*1.6726e-27*1e6*n[indx]*1e9 ; Ram pressure (nPa)
  parmod[1]=0. ;DST
  parmod[2]=by[indx] ;IMF By
  parmod[3]=bz[indx] ;IMF Bz
  
  if n[indx] lt -1 then begin ; if statement for datagap
    parmod[*]=0.
    parmod[0]=2.5 ; Ram pressure (nPa)
    parmod[1]=0. ;DST
    parmod[2]=0 ;IMF By
    parmod[3]=-1 ;IMF Bz
  endif
  

  t = time_struct(time[indx])
  geopack_recalc, t.year, t.month,t.date, t.hour, t.min, t.sec,TILT=TILT, /date
  
  popen, '/Users/bmwalsh/Documents/Research/DXL/DXLIII/upload_files/Tsy_cusp',encapsulated=1, xsize = 4, ysize = 2.75
;  ;X Z Plot
;  ;T96 model
  plot,[10,-5],[10,-14],/nodata,/isotropic,XTITLE='X!LGSM!N[Re]',YTITLE='Z!LGSM!N[Re]',XRANGE=[15,-27], $
    YRANGE=[-15,15], /noerase, charsize = 0.7
    
  loadct, 1  
  FOR i=0,40,5 DO BEGIN
    geopack_sphcar,1,i,0,x,y,z,/degree,/to_rect
    geopack_t96, parmod, x,y,z,bx,by,bz
    geopack_trace,x,y,z,1,parmod,xf,yf,zf,fline=fline,/T96, TILT=TILT
    oplot,fline[*,0],fline[*,2],color=220, THICK=3

    geopack_sphcar,1,i,180,x,y,z,/degree,/to_rect
    geopack_t96, parmod, x,y,z,bx,by,bz
    geopack_trace,x,y,z,1,parmod,xf,yf,zf,fline=fline,/T96, TILT=TILT
    oplot,fline[*,0],fline[*,2],color=220, THICK=3
  ENDFOR

  FOR i=50, 90, 5 DO BEGIN
    geopack_sphcar,1,i+90,0,x,y,z,/degree,/to_rect
    geopack_t96, parmod, x,y,z,bx,by,bz
    geopack_trace,x,y,z,-1,parmod,xf,yf,zf,fline=fline,/T96,TILT=TILT
    oplot,fline[*,0],fline[*,2],color=220, THICK=3

    geopack_sphcar,1,i+90,180,x,y,z,/degree,/to_rect
    geopack_t96, parmod, x,y,z,bx,by,bz
    geopack_trace,x,y,z,-1,parmod,xf,yf,zf,fline=fline,/T96,TILT=TILT
    oplot,fline[*,0],fline[*,2],color=220, THICK=3
  ENDFOR
  
  loadct, 40
  ;;Plot earth
  radius=1
  points = (2*!PI / 99.0) * FINDGEN(100)
  x = radius * COS(points)
  y = radius * SIN(points)
  oplot, x,y, thick = 5

  xyouts, -6,-13, 'Real-Time TS04 Model', /data, charsize = 0.6
  xyouts, 14,-13, 'Dipole Tilt:'+trim(tilt,'(F7.3)')+' Deg', /data, charsize = 0.6
  xyouts, 90,90,'Plot Generated: '+systime()+',  Tsyganenko Magnetic Field Model', /device, charsize = 0.4
  if n[indx] lt -1 then begin ; if statement for datagap
    xyouts, 14,13, 'DSCOVR Datagap. Using default values:', /data, charsize = 0.6
    xyouts, 14,11, '[Bz=1nT, By=0nT, Dst=0nT, Pdyn=2.5nPa]', /data, charsize = 0.6
  endif
  
  
  pclose
  print, 'Completed making plots'
  
  
  ;==================== Make Go/No Go plot =================
    
  popen, '/Users/bmwalsh/Documents/Research/DXL/DXLIII/upload_files/Cusp_predict',encapsulated=1,xsize = 6, ysize = 7
  
  by_const_high = fltarr(n_elements(time))
  by_const_low = fltarr(n_elements(time))
  by_const_high[*] = 7
  by_const_low[*] = -7
  store_data,'By_high',dat={x:time, y:by_const_high}
  store_data,'By_low',dat={x:time, y:by_const_low}

  store_data,'By_plot',data=['By_high','By','By_low']
  options, 'By_plot',constant=0,colors=['r','b','r'],labels=['By_min','By','By_max'],$
    labflag=1,yrange = [-12,12], ytitle = 'By GSM', ysubtitle = '[nT]', $
    psym = symcat(16), symsize = 0.4  
    
  ; prep Bz
  bz_const_high = fltarr(n_elements(time))
  bz_const_low = fltarr(n_elements(time))
  bz_const_high[*] = 12
  bz_const_low[*] = -9
  store_data,'Bz_high',dat={x:time, y:bz_const_high}
  store_data,'Bz_low',dat={x:time, y:bz_const_low}

  store_data,'Bz_plot',data=['Bz_high','Bz','Bz_low']
  options, 'Bz_plot',constant=0,colors=['r','b','r'],labels=['Bz_min','Bz','Bz_max'],$
    labflag=1,yrange = [-14,14], ytitle = 'Bz GSM', ysubtitle = '[nT]', $
    psym = symcat(16), symsize = 0.4
    
  store_data,'flux',dat={x:time, y:flux}
  flux_limit = fltarr(n_elements(time))
  flux_limit[*] = 2.8
  store_data,'flux_limit',dat={x:time, y:flux_limit}
  store_data,'flux_plot',data=['flux','flux_limit']
  indx = where((flux lt 100) and (flux gt 0),ct)
  maxv = round(max(flux[indx])+2.)
  if ct lt 2 then maxv=10 ; if statement if there is no data
  
  options, 'flux_plot', yrange = [0, maxv], ytitle = 'Flux', ysubtitle = '10!U8!N [1/cm!U-2!N*s]', $
    psym = symcat(16), symsize = 0.5,colors=['b','r'],labels=['Flux','Bottom_Limit'],labflag=1, constant=0

  tplot, ['By_plot','Bz_plot','flux_plot']
  timebar,max(time)-35.*60.
  xyouts, 100,100,'Plot Generated:'+systime(), /device
    
  pclose
  ;================= Make parts for the movie =================
  
  
  
  ;========================Cusp Footprint==========================


  ; cusp position or radial distance from Earth in RE
  rho = 8.0 ; altitude in re
  ;------------------------------

  ; Dont change these ======================
  ; Constants from Tsyganenko and Russell JGR 1999
  phi_c0 = 0.24
  alpha1 = 0.1287
  alpha2 = 0.0314
  ;=====================================

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

  ; create tplot var
  indx = closest(time,max(time)-35.*60.)
  time_cusp = [time[indx]]
  pos = fltarr(1,3)
  pos[0,0] = x_sm
  pos[0,1] = y_sm
  pos[0,2] = z_sm
  

  ; Get cusp position in ionosphere
  trace2iono,time_cusp,pos,out_foot_array,external_model='t96',in_coord='sm',out_coord='geo',par = parmod
  xyz_to_polar,out_foot_array,mag=mag_val,theta=theta_val,phi=phi_val

  GEOPACK_CONV_COORD, 1,0,0,xgeo,ygeo,zgeo, /from_gse, /to_geo; Find local noon
  xyz_to_polar,[xgeo,ygeo,zgeo],mag=mag_val,theta=theta_noon,phi=phi_noon ; convert local noon to polar coord

  
  
;stop
  thm_map_set,central_lon=180+phi_noon,central_lat=90,xsize=700,ysize=700,scale = 5e7
  thm_map_add,geographic_lats=45+5*indgen(10),geographic_lons=0+findgen(20)*20,geographic_linestyle=2 ; add grid lines

  ;ionospheric footpoint trace plotted on the map
  loadct, 40
  
  plots,phi_val,theta_val, psym=symcat(16), symsize = 5, color = 50

  plots,[(phi_noon),(phi_noon)],[90,10],color = 250,thick=5 ; Sun line
  plots,[-147.447222],[65.136666], psym=symcat(14), symsize = 4, color = 250 ; Poker flat
  plots,[15.64689],[78.22334], psym=symcat(14), symsize = 4, color = 250 ; EISCAT

  xyouts, 340,600,'Local Noon',color=250, charsize = 1.5,/device, ORIENTATION=90
  xyouts, 10,10,'Plot Generated:'+systime(),color=250, charsize = 1.5,/device
  xyouts, 10,35,'Real-Time Cusp Footprint. TS04 Model. DSCOVR SW data.',color=250, charsize = 1.5,/device
  xyouts, 10,60,'Dipole Tilt: '+trim(tilt,'(F7.1)')+' degrees',color=250, charsize = 1.5,/device

  xyouts, phi_val-10,theta_val-2,'Cusp Footprint',color=250, charsize = 2.0,/data

  if n[indx] lt -1 then begin ; if statement for datagap
    xyouts, 10,675,'DSCOVR Datagap. Using default SW values.', /device, color = 250,charsize=1.5
  endif

;  device,set_resolution=[600,600]
;  tvlct,red,green,blue,/get ; read current palette

  filename='/Users/bmwalsh/Documents/Research/DXL/DXLIII/upload_files/Cusp_footprint.png'
  write_png, filename,tvrd(),red,green,blue    
  erase
  store_data,'*',/delete
  
  ; ================ Done plotting ================
  
;  stop
  wait, 55
 endwhile
end