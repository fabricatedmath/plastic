#pragma once

#define WFFINITMIN 0.0
#define WFFINITMAX 0.1
#define MAXW 50.0

#define NBE 2
#define NBI 1

#define NBNEUR (NBE + NBI)

#define NUMTHREADS 128

#define PATCHSIZE 17
#define FFRFSIZE (2 * PATCHSIZE * PATCHSIZE)

#define LATCONNMULTINIT 5.0  // The ALat coefficient; (12 * 36/100)
#define LATCONNMULT LATCONNMULTINIT

#define WEI_MAX (20.0 * 4.32 / LATCONNMULT) //1.5
#define WIE_MAX (.5 * 4.32 / LATCONNMULT)
#define WII_MAX (.5 * 4.32 / LATCONNMULT)

#define DT 1.0
#define INPUTMULT ((DT / 1000.0) * 150.0 * 2.0)

#define NBSTEPSPERPRES 350
#define TIMEZEROINPUT 100
#define NBSTEPSSTIM (NBSTEPSPERPRES - TIMEZEROINPUT)

#define VSTIM 1.0

#define GLEAK 30.0
#define ELEAK -70.6

#define IZHREST -70.5
#define THETAVLONGTRACE -45.3
#define THETAVPOS -45.3
#define THETAVNEG ELEAK
#define VRESET ELEAK
#define VTREST -50.4
#define VTMAX -30.4
#define VPEAK 20
#define MINV -80.0
#define BASEALTD (14e-5 * 1.5 * 1.0)
#define RANDALTD 0.0

#define DELTAT 2.0
#define INVDELTAT (1 / DELTAT)
#define CONSTA 4
#define CONSTB 0.0805
#define CONSTC 281.0
#define INVCONSTC (1 / CONSTC)
#define ISP 400.0

#define NBSPIKINGSTEPS 1

#define NEGNOISERATE 0.0
#define POSNOISERATE 1.8

#define TAUADAP 144.0
#define INVTAUADAP (1 / TAUADAP)

#define TAUZ 40.0
#define INVTAUZ (1 / TAUZ)

#define TAUVTHRESH 50.0
#define INVTAUVTHRESH (1 / TAUVTHRESH)

#define TAUVNEG 10.0
#define INVTAUVNEG (1 / TAUVNEG)

#define TAUVPOS 7.0
#define INVTAUVPOS (1 / TAUVPOS)

#define TAUVLONGTRACE 20000.0
#define INVTAUVLONGTRACE (1 / TAUVLONGTRACE)

#define TAUXPLAST 15.0
#define INVTAUXPLAST (1 / TAUXPLAST)

#define VREF2 50.0
#define INVVREF2 (1 / VREF2)

#define ALTP (8e-5 * 0.008 * 1.0)
#define ALTPMULT 0.75

#define WPENSCALE 0.33
