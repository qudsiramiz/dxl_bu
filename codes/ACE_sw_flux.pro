; BMW 2015

thm_init

!Y.STYLE = 1

limit = 0
; repeat every 1 minutes
while limit lt 1 do begin

  ;1. grab sw data from web
  spawn, 'python /Users/bmwalsh/Documents/Research/DXL/download.py'
  
  ;2. read in file
  readcol, '/Users/bmwalsh/Documents/Research/DXL/ace-swepam.txt', skipline = 18, year, month, day, hhmm, day0,day1, $
    seconds, n, speed, temp, format='(I,I,I,A,I,I,I,F,F,F)'
    
  minute = float(strmid(hhmm,2))
  hour = float(strmid(hhmm,0,2))
  
  time = julday(month,day,year, hour,minute, seconds)
  time = cluster2Th_time(time)
  
  
  flux = speed*n*1e-3
  
  store_data,'flux',dat={x:time, y:flux}
indx = where((flux lt 100) and (flux gt 0))
maxv = round(max(flux[indx])+2.)
  options, 'flux', yrange = [0, maxv], ytitle = 'Flux', ysubtitle = '10!U8!N [1/cm!U-2!N*s]', $
    psym = symcat(16), symsize = 0.5, panel_size=1.75,constant = 5
  
  
  indx = where((n lt 100) and (n gt 0.01))
  maxv = round(max(n[indx])+10.)
  store_data,'density',dat={x:time, y:n}
  options, 'density', /ylog, colors = ['r'], ytitle = 'n', ysubtitle = '[cm!U-3!N]', $
    psym = symcat(16), symsize = 0.5, yrange = [0.5,maxv]


  indx = where((speed gt 100) and (flux lt 990))
  maxv = (round(max(speed[indx])/100.)+1.0)*100.
  minv = (round(min(speed[indx])/100.)-1.0)*100.
   
  store_data,'speed',dat={x:time, y:speed}
  options, 'speed', yrange = [minv,maxv], colors = ['b'], ytitle = 'Bulk Speed', $
    ysubtitle = '[km/s]', psym = symcat(16), symsize = 0.5
  
  store_data,'temp',dat={x:time, y:temp}
  
  
; read in mag data
readcol, '/Users/bmwalsh/Documents/Research/DXL/ace-magnetometer.txt', skipline = 20, year, month, day, hhmm, day0,day1, $
  seconds, bx, by, bz,bt,lat,lon, format='(I,I,I,A,I,I,I,F,F,F,F,F,F)'  

store_data,'Bx',dat={x:time, y:bx}
store_data,'By',dat={x:time, y:by}
store_data,'Bz',dat={x:time, y:bz}

store_data,'B',data=['Bx','By','Bz']
options, 'B',constant=0,colors=['b','g','r'],labels=['Bx','By','Bz'],$
  labflag=1,yrange = [-10,10], ytitle = 'B GSM', ysubtitle = '[nT]', $
  psym = symcat(16), symsize = 0.4

  
  timespan, time_string(min(time)),2,/hours
  
  tplot_options,'charsize',1.1
  tplot_options,'thick',6.0
  tplot_options,'title','2 Hours ACE, Real-Time'

  ;tplot_options,'font',2.0
  
  
  tplot, ['B','density','speed','flux']
  timebar,max(time)-35.*60.
  
  
  ; print to file
  set_plot,'z'
  
  !P.BACKGROUND=255
  !P.COLOR=0
  
  device,set_resolution=[600,600]
  tvlct,red,green,blue,/get ; read current palette
  
  filename='/Users/bmwalsh/Documents/Research/DXL/SW_ACE.png'
  write_png, filename,tvrd(),red,green,blue
  
  wait, 60.

  ;3. Send to server
  spawn, 'python /Users/bmwalsh/Documents/Research/DXL/post_file.py'

endwhile

end