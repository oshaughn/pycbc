#
# matchedfiltertwosided.py
#    matchedfilter applied to complex-valued timeseries (i.e., frequencyseries has twosided==True)


import numpy 
from math import sqrt
from pycbc.types import TimeSeries, FrequencySeries, zeros, Array
from pycbc.types import complex_same_precision_as, real_same_precision_as
from pycbc.fft import fft, ifft, IFFT
import pycbc.scheme
from pycbc import events
import pycbc

from pycbc.filter.matchedfilter import get_cutoff_indices


def make_frequency_series_twosided(vec):
    """Return a frequency series of the input vector.

    If the input is a frequency series it is returned, else if the input
    vector is a real time series it is fourier transformed and returned as a
    frequency series.

    Parameters
    ----------
    vector : TimeSeries or FrequencySeries

    Returns
    -------
    Frequency Series: FrequencySeries
        A frequency domain version of the input vector.
        Forces twosided=true
    """
    if isinstance(vec, FrequencySeries):
        return vec
    if isinstance(vec, TimeSeries):
        N = len(vec)
        n = N
        delta_f = 1.0 / N / vec.delta_t
        vectilde =  FrequencySeries(zeros(n, dtype=complex_same_precision_as(vec)),
                                    delta_f=delta_f, copy=False, twosided=True)
        fft(vec, vectilde)
        return vectilde
    else:
        raise TypeError("Can only convert a TimeSeries to a FrequencySeries")


def matched_filter_twosided_core(template, data, psd=None, low_frequency_cutoff=None,
                  high_frequency_cutoff=None, h_norm=None, out=None, corr_out=None):
    """ Return the complex snr and normalization.

    Return the complex snr, along with its associated normalization of the template,
    matched filtered against the data.

    Parameters
    ----------
    template : TimeSeries or FrequencySeries
        The template waveform.  Use only if dtype=complex or twosided
    data : TimeSeries or FrequencySeries
        The strain data to be filtered.
    psd : {FrequencySeries}, optional
        The noise weighting of the filter.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin the filter calculation. If None, begin at the
        first frequency after DC.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop the filter calculation. If None, continue to the
        the nyquist frequency.
    h_norm : {None, float}, optional
        The template normalization. If none, this value is calculated internally.
    out : {None, Array}, optional
        An array to use as memory for snr storage. If None, memory is allocated
        internally.
    corr_out : {None, Array}, optional
        An array to use as memory for correlation storage. If None, memory is allocated
        internally. If provided, management of the vector is handled externally by the
        caller. No zero'ing is done internally.

    Returns
    -------
    snr : TimeSeries
        A time series containing the complex snr.
    corrrelation: FrequencySeries
        A frequency series containing the correlation vector.
    norm : float
        The normalization of the complex snr.
    """
    htilde = make_frequency_series_twosided(template)
    stilde = make_frequency_series_twosided(data)

    if len(htilde) != len(stilde):
        raise ValueError("Length of template and data must match")

    N = len(stilde)
    # can't just use two limits : need to base criteria on frequency value array
    # implementation will ONLY work for numpy arrays, where boolean indexing is valid
    fvals = htilde.sample_frequencies
    indx_ok = numpy.logical_and( numpy.abs(fvals) >low_frequency_cutoff, numpy.abs(fvals) < high_frequency_cutoff)

    if corr_out is not None:
        qtilde = corr_out
    else:
        qtilde = zeros(N, dtype=complex_same_precision_as(data))

    if out is None:
        _q = zeros(N, dtype=complex_same_precision_as(data))
    elif (len(out) == N) and type(out) is Array and out.kind =='complex':
        _q = out
    else:
        raise TypeError('Invalid Output Vector: wrong length or dtype')

    correlate(htilde[indx_ok], stilde[indx_ok], qtilde[indx_ok])

    if psd is not None:
        if isinstance(psd, FrequencySeries):
            if psd.delta_f == stilde.delta_f :
                qtilde[indx_ok] /= psd[indx_ok]
            else:
                raise TypeError("PSD delta_f does not match data")
        else:
            raise TypeError("PSD must be a FrequencySeries")

    ifft(qtilde, _q)

    if h_norm is None:
        h_norm = sigmasq_twosided(htilde, psd, low_frequency_cutoff, high_frequency_cutoff)

    norm = (2.0 * stilde.delta_f) / sqrt( h_norm)
    delta_t = 1.0 / (N * stilde.delta_f)

    return (TimeSeries(_q, epoch=stilde._epoch, delta_t=delta_t, copy=False),
           FrequencySeries(qtilde, epoch=stilde._epoch, delta_f=htilde.delta_f, copy=False),
           norm)



def sigmasq_twosided(htilde, psd = None, low_frequency_cutoff=None,
            high_frequency_cutoff=None):
    """Return the loudness of the waveform. This is defined (see Duncan
    Brown's thesis) as the unnormalized matched-filter of the input waveform,
    htilde, with itself. This quantity is usually referred to as (sigma)^2
    and is then used to normalize matched-filters with the data.

    Parameters
    ----------
    htilde : TimeSeries or FrequencySeries
        The input vector containing a waveform.  twosided
    psd : {None, FrequencySeries}, optional
        The psd used to weight the accumulated power.  Must also be twosided.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin considering waveform power.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop considering waveform power.

    Returns
    -------
    sigmasq: float
    """
    htilde_new = make_frequency_series_twosided(htilde)
    htilde=htilde_new
    N = len(htilde)
    norm = 2.0 * htilde.delta_f
    # Only works with numpy array datatypes for ht
    fvals = htilde.sample_frequencies
    # Positive frequencies : identified by usual method
    kmin_onesided, kmax_onesided = get_cutoff_indices(low_frequency_cutoff, high_frequency_cutoff, htilde.delta_f, N)
    # Negative frequencies: use known packing
    kmax_plus, kmin_plus = [N-kmin_onesided, N-kmax_onesided]  # reverse order

    ht = htilde[kmin_onesided:kmax_onesided]
    ht_minus = htilde[kmin_plus:kmax_plus]

    if psd:
        try:
            numpy.testing.assert_almost_equal(ht.delta_f, psd.delta_f)
        except:
            raise ValueError('Waveform does not have same delta_f as psd')

    if psd is None:
        sq = ht.inner(ht)
        sq_minus = ht.inner(ht_minus)
    else:
        # PSD structure also needs to be twosided
        sq = ht.weighted_inner(ht, psd[kmin_onesided:kmax_onesided])
        sq_minus = ht.weighted_inner(ht_minus, psd[kmin_plus:kmax_plus])

    return (sq.real + sq_minus.real) * norm

def sigma_twosided(htilde, psd = None, low_frequency_cutoff=None,
        high_frequency_cutoff=None):
    """ Return the sigma of the waveform. See sigmasq for more details.

    Parameters
    ----------
    htilde : TimeSeries or FrequencySeries
        The input vector containing a waveform.  twosided
    psd : {None, FrequencySeries}, optional
        The psd used to weight the accumulated power.  Must also be twosided.
    low_frequency_cutoff : {None, float}, optional
        The frequency to begin considering waveform power.
    high_frequency_cutoff : {None, float}, optional
        The frequency to stop considering waveform power.

    Returns
    -------
    sigmasq: float
    """
    return sqrt(sigmasq_twosided(htilde, psd, low_frequency_cutoff, high_frequency_cutoff))
