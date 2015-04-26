import numpy as np
cimport numpy as np

def sint(float x, np.ndarray f):

    assert f.dtype == float

    cdef float dampfac = 3.25
    cdef int ksize   = 21

    cdef int nf = f.shape[0]
    cdef int ix = int(x)
    cdef float fx = x - ix

    if fx==0.:
        # There are integral values of the independent variable. Select the function
        # values directly for these points.
        return f[ix]

    # Use sinc interpolation for the points having fractional values.
    xkernel=np.zeros(ksize)
    vals=np.zeros(ksize)
    xkernel = ( np.arange( ksize,dtype=float ) - 10 ) - fx
    u1 = xkernel / dampfac
    u2 = np.pi * xkernel
    sinc = np.exp( -( u1*u1 ) ) * np.sin( u2 ) / u2
    lobe = ( np.arange( ksize,dtype=int ) - 10 ) + ix
    for j in range(ksize):
        if lobe[j]<0: vals[j] = f[0]
        elif lobe[j]>=nf: vals[j] = f[-1]
        else: vals[j] = f[lobe[j]]
    return np.dot( sinc , vals )

def sint2d(x, y, np.ndarray f):

    # Get size of input array.
    assert f.dtype == float
    cdef int xsize = f.shape[0]
    cdef int ysize = f.shape[1]
    if x<0 or y<0 or x>xsize-1 or y>ysize-1: return 0

    # Half-size of the kernel.
    cdef int delta = 10

    # Compute integer and fractional parts of input position.
    ix = int( x )
    fx = x - ix
    iy = int( y )
    fy = y - iy

    yoff = min(iy, delta)
    ly   = iy - yoff
    hy   = iy + yoff
    if hy>=ysize: hy = ysize-1
    ny   = max(2,hy - ly + 1)

    vals = np.zeros( ny )
    for j in range(ny):
        xoff = min(ix, delta)
        lx = ix - xoff
        hx = ix + xoff
        if hx>=xsize: hx = xsize-1
        if hx-lx+1<2: hx+=1
        r1 = f[ lx : hx+1, ly+j ]
        x1 = fx + xoff
        vals[j] = sint( x1, r1 )
    return sint( fy+yoff, vals )
