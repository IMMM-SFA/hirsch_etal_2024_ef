# SP2016.ctl
#
  South Platte StateMod Historical Monthly Model (SP2016)     
  May 2017 
    1950     : iystr   STARTING YEAR OF SIMULATION
    2012     : iyend   ENDING YEAR OF SIMULATION
       2     : iresop  OUTPUT UNIT OPTION. 1 FOR [CFS], 2 FOR [AF], 3 FOR [KAF]
       1     : moneva  TYPE OF EVAP. DATA. 0 FOR VARIANT DATA. 1 FOR CONS. DATA
       1     : ipflo   TYPE OF STREAM INFLOW. 1 FOR TOTAL FLOW. 2 FOR GAINS
       0     : numpre  NO. OF PRECIPITATION STATIONS
      23     : numeva  NO. OF EVAPORATION STATIONS
      -1     : interv  NO. OF TIME INTERVALS IN DELAY TABLE. MAXIMUM=60.
  1.9835     : factor  FACTOR TO CONVERT CFS TO AC-FT/DAY (1.9835)
  1.9835     : rfacto  DIVISOR FOR STREAM FLOW DATA;    ENTER 0 FOR DATA IN cfs, ENTER 1.9835 FOR DATA IN af/mo
  1.9835     : dfacto  DIVISOR FOR DIVERSION DATA;      ENTER 0 FOR DATA IN cfs, ENTER 1.9835 FOR DATA IN af/mo
  0          : ffacto  DIVISOR FOR IN-STREAM FLOW DATA; ENTER 0 FOR DATA IN cfs, ENTER 1.9835 FOR DATA IN af/mo
  1.0        : cfacto  FACTOR TO CONVERT RESERVOIR CONTENT TO AC-FT
  0.0833     : efacto  FACTOR TO CONVERT EVAPORATION DATA TO FEET
  0.0833     : pfacto  FACTOR TO CONVERT PRECIPITATION DATA TO FEET
  CYR        : cyr1    Year type (a5 right justified !!)
       1     : icondem    1=Historic Demand, 2=Historic Sum, 3=Structure Demand, 4=Supply Demand, 5=Decreed Demand
       0     : Detailed output  0 = off, 1=print river network,  -n= detailed printout, 100+ operating rule, 200+standard rule
       1     : ireopx  Re-operation switch (0=re-operate;1=no re-operation: -10=10 af/mon=0.63 cfs)
       1     : ireach  0=no instream reach; 1=yes instream flow reach
       0     : icall   0=no detailed call info., 1=yes detailed call info
       0     : water right where detailed call data is requested
       0     : iday    0=monthly model; 1=daily
       1     : iwell   0=no wells & not in *.rsp file; 1=yes wells; -1=no wells but in *.rsp file
       0     : gwmaxrc Constant Maximum stream loss (cfs). Only used if iwell = 2
       0     : isjrip  San Juan RIP
      10     : itsfile -1 skip *.tsp, 0=no tsfile, 1=variable n, 10 variable n, well area, capacity, etc.
       1     : ieffmax -1 skip *.iwr, 0 no *.iwr, 1 yes *.iwr, 2=read but use ave n
       2     : isprink 0=off, 1=Maximum Supply, 2=Mutual Supply
       3     : soild   Soil Moisture 0 = off, +n = root depth
       0     : isigfig 0=none, 1=one