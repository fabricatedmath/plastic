#pragma once

#include "state.h"

template<typename F,typename I>
void wrapper(MutableState<F,I> mutableState, StaticState<F,I> staticState, Buffers<F,I> buffers);
