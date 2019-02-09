#pragma once

#include "state.h"

template<typename F,typename I>
MutableState<F,I> run(MutableState<F,I> mutableState, StaticState<F,I> staticState, Buffers<F,I> buffers);

template<typename F,typename I>
MutableState<F,I> run(MutableState<F,I> mutableState, StaticState<F,I> staticState, Buffers<F,I> buffers, RandomHistorical<F> randomHistorical);
