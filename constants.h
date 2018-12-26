#pragma once

#define WFFINITMIN 0.0
#define WFFINITMAX 0.1
#define MAXW 50

#define NBE 100
#define NBI 20

#define NBNEUR (NBE + NBI)

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
