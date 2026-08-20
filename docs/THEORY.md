# Theory

This is the long-form companion to the code: what the engine is doing, why, and
where the model stops being true. It is written to be read start to finish by
someone who knows signal processing but not ultrasound.

**Contents**

1. [The physical picture](#1-the-physical-picture)
2. [The forward model](#2-the-forward-model)
3. [Transmit: delay laws and the `t = 0` convention](#3-transmit-delay-laws-and-the-t--0-convention)
4. [Receive: why baseband IQ](#4-receive-why-baseband-iq)
5. [Beamforming](#5-beamforming)
6. [Resolution, contrast, and the trade between them](#6-resolution-contrast-and-the-trade-between-them)
7. [Judging a beamformer](#7-judging-a-beamformer)
8. [Doppler](#8-doppler)
9. [Elastography](#9-elastography)
10. [What is not modelled](#10-what-is-not-modelled)

---

## 1. The physical picture

An ultrasound probe is an array of piezoelectric elements. Apply a voltage and
an element vibrates, launching a pressure pulse into the tissue; the same element
converts returning pressure back into voltage. Everything else is timing.

Three numbers set the scale of the entire subject:

- **Speed of sound**, about **1540 m/s** in soft tissue. Sound is roughly 200,000
  times slower than light, which is why time-of-flight imaging is practical here
  and not in optics: an echo from 5 cm deep returns in 65 µs, comfortably
  measurable with ordinary electronics.
- **Wavelength**, λ = c / f₀. At 5 MHz that is **308 µm**. Resolution is measured
  in wavelengths, so this is the yardstick for everything spatial.
- **Attenuation**, roughly **0.5 dB/(cm·MHz)** one way. Round trip at 5 MHz that
  is 5 dB per centimetre of depth. This single number is why ultrasound is a
  trade-off machine: higher frequency gives finer resolution and less penetration,
  and there is no way around it.

The last one deserves emphasis. A 10 MHz probe resolves twice as finely as a
5 MHz probe and loses signal twice as fast with depth. Every clinical probe
choice is a position on that line, and no amount of signal processing moves it.

### Why an array

A single element transmits a beam it cannot steer and receives from everywhere
that beam went. An array of them can do two things a single element cannot:

- **Steer**, by firing elements at staggered times so the wavefronts add up along
  a chosen direction;
- **Focus**, on both transmit and receive, by choosing those delays so that all
  contributions arrive at one point simultaneously.

Receive focusing is the more powerful of the two, because it can be done
*retrospectively and differently for every pixel* — the data are already
recorded, so nothing stops you from applying the delay law for a point at 10 mm
and then re-applying a different one for a point at 50 mm. Transmit focusing has
to be committed to before the pulse is fired.

---

## 2. The forward model

The simulator (`core/src/simulator.cpp`) treats tissue as a cloud of independent
point scatterers. Each one re-radiates whatever pressure reaches it, echoes
superpose linearly, and there is no multiple scattering. That sounds crude, and
it produces astonishingly realistic images — because the thing that dominates
ultrasound's appearance is not the individual scatterers but their *interference*.

### Speckle

Put more than about ten scatterers inside one resolution cell and their echoes
add with random relative phases. The resulting envelope is Rayleigh-distributed,
with the characteristic ratio

$$\frac{\mu}{\sigma} = \frac{\sqrt{\pi}/2}{\sqrt{1 - \pi/4}} \approx 1.91$$

This granular texture is **speckle**, and it is the most misunderstood thing in
the field. It is *not noise*. It is a deterministic interference pattern:
image the same tissue twice without moving and you get the identical speckle
both times. That determinism is what makes speckle *tracking* possible, which is
what Doppler and elastography are built on. A "denoising" step that averages
speckle away is destroying the signal those methods need.

The engine measures this: `usx.metrics.speckle_snr` reports µ/σ, and a value near
1.91 confirms a phantom is fully developed. A much *higher* value is the warning
sign — it means a beamformer has smoothed the texture out.

### The two-stage simulation

For each transmit event and each scatterer:

1. **Incident field.** Sum the contribution of every firing element at the
   scatterer's position, each delayed by its own transmit delay plus its own
   propagation time, and weighted by aperture apodization, element directivity,
   spherical spreading and attenuation. Because each element is summed
   separately, the result is a genuinely diffracted field — it has a focal zone,
   edge waves and transmit sidelobes.
2. **Received echo.** The scatterer re-radiates that waveform; each receive
   element sees a delayed, attenuated, direction-weighted copy of it.

The per-scatterer incident waveform is trimmed to its significant support before
stage 2, since everything in stage 2 costs time proportional to the length kept.
That trim threshold (`SimulatorConfig::incident_trim_db`) is the main
speed/accuracy dial.

### Why the simulator does not reuse the beamformer's model

The beamformer assumes an idealized wavefront — a perfect plane, or a perfect
spherical shell converging on the focus. The simulator does not; it sums the real
per-element contributions.

This asymmetry is deliberate and it is the reason the tests mean anything. If
both sides used the same idealization, beamforming simulated data would confirm
itself: the delays would cancel by construction and a sign error in the
wavefront model would be invisible. Because the simulator's field is built from
different arithmetic, a point target reconstructing at the right coordinates is
real evidence that the delay model is right.

### Element directivity

A finite-width element is not an omnidirectional point. Its far-field amplitude
response is

$$D(\theta) = \cos\theta \cdot \mathrm{sinc}\!\left(\frac{\pi w \sin\theta}{\lambda}\right)$$

the sinc being the Fourier transform of the element's uniform aperture and the
cosine an obliquity factor. This is why fine-pitch arrays image wide sectors and
coarse ones cannot, and it is a hard limit: no beamformer recovers energy the
element never received.

---

## 3. Transmit: delay laws and the `t = 0` convention

### The three transmit geometries

**Focused.** Choose delays so every element's contribution reaches a focal point
`F` simultaneously. The element furthest from `F` has the longest path, so it
fires first:

$$\tau_i = \frac{|F - O| - |F - p_i|}{c}$$

where `O` is the beam origin on the array face. One transmit forms one scanline;
a 128-line image costs 128 transmits, and the frame rate is capped at
`c / (2 · depth · n_lines)` — about 40 Hz at 5 cm with 128 lines.

**Plane wave.** A flat wavefront with normal `(sin α, 0, cos α)`, produced by a
linear delay ramp:

$$\tau_i = \frac{(x_i - x_O)\sin\alpha}{c}$$

One transmit insonifies the *entire* field of view, so one transmit reconstructs
an entire image and the frame rate limit becomes `c / (2 · depth)` — about 15 kHz
at 5 cm. This is "ultrafast" imaging, and it is what makes shear-wave
elastography and functional ultrasound possible.

**Diverging.** Time-reversed focusing: a virtual source *behind* the array, so
the beam spreads out to cover a sector. This is how phased arrays do ultrafast
imaging.

$$\tau_i = \frac{|p_i - F| - |O - F|}{c}, \qquad F_z < 0$$

### Coherent compounding

A single plane wave gives a bad image, because there is no transmit focusing at
all — the two-way point spread function is just the receive beam. The fix is to
fire several plane waves at different angles and **coherently sum** the
beamformed complex images. Each angle illuminates the target from a different
direction, so summing them synthesises a transmit focus at every depth
simultaneously — better than a single focused transmit, which is only in focus
near its focal zone.

The engine's `test_resolution_improves_with_more_angles` measures this: eleven
angles narrow the beam by more than 25 % versus one.

The useful angular span is about `±arctan(1/(2·F#))` of the receive f-number.
Beyond that the extra transmits add little new spatial-frequency content and
mostly contribute grating-lobe energy.

### The `t = 0` convention

This is the single most common source of a beamformed image that is uniformly,
plausibly slightly out of focus — and the reason this engine defines it
explicitly and tests it.

**`t = 0` is the instant the transmitted wavefront passes through the beam's
origin on the transducer face.**

Every delay law above is written to make that true, which means some delays come
out negative. Real transmit electronics cannot fire before they start, so
hardware fires element `i` at `delays[i] - t0_offset` on its own clock, where
`t0_offset = min(delays)`; the whole acquisition is then shifted by that amount.
`compute_transmit_delays` returns it for exactly this reason.

Why bother? Because it makes the beamformer's job stateless. Given the
convention, the round-trip time to a pixel is

$$\tau(\mathbf r) = \tau_{\text{tx}}(\mathbf r) + \frac{|\mathbf r - p_i|}{c}$$

with no per-event bookkeeping, and the simulator and the beamformer share a clock
by construction. When adapting external data, the one thing that must be got
right is where that recording's sample 0 sits relative to this definition; set
`Acquisition.t0` accordingly and everything else follows.

The focused case has one more subtlety. Ahead of the focus the wavefront is
*converging*, so distance travelled from the origin increases as `|r − F|`
*decreases*; past the focus it diverges again:

$$\tau_{\text{tx}}(\mathbf r) = \frac{|F - O| \pm |\mathbf r - F|}{c}$$

with the sign determined by which side of the focus the point is on. Getting that
sign wrong gives an image correctly focused in a band and mirrored beyond it.

---

## 4. Receive: why baseband IQ

Raw RF is sampled at four or more times f₀, but the information occupies only the
transducer's bandwidth around f₀. Mixing that band to DC and low-pass filtering
gives complex baseband **IQ**:

$$z(t) = \mathrm{LP}\left\{ s(t)\, e^{-j 2\pi f_{\text{dem}} t} \right\}$$

The obvious benefit is data rate. The real benefit is interpolation.

A beamformer must resample every channel at an arbitrary, continuously varying
delay. On RF you are interpolating a 5 MHz carrier: linear interpolation between
samples introduces phase error unless you oversample heavily or use an expensive
interpolator, and phase error across the aperture is exactly what destroys the
coherent sum. On IQ the carrier is gone, what remains varies on the scale of the
pulse envelope, cheap linear interpolation is accurate, and the sub-sample delay
becomes an *exact* phase rotation rather than an approximation. Every real-time
system does its beamforming this way.

The price is bookkeeping: after mixing, a delay `τ` carries a factor
`e^{−j2πf_dem τ}` that the beamformer must undo (§5). Forget it and the channels
align in envelope but not in phase, and they cancel instead of summing — the
diagnostic symptom being an image that gets *darker* as the aperture opens.

Two implementation details that are easy to get wrong and are handled explicitly
in `core/src/demodulate.cpp`:

- The mixer uses **absolute acquisition time**, not sample index, so events with
  different `t0` share a phase reference.
- The FIR's linear-phase **group delay is compensated**, so IQ sample `k` still
  corresponds to `t0 + k·dec/fs`. Skipping this shifts every echo axially by half
  the filter length, which reads as a depth calibration error.

---

## 5. Beamforming

For each pixel, ask: if a scatterer were *here*, when would its echo have reached
each element? Read each channel at that time, so the wanted echo aligns across
all channels while everything else does not, and combine.

$$y_i(\mathbf r) = w_i(\mathbf r)\, z_i\!\left(\tau_{\text{tx}}(\mathbf r) + \frac{|\mathbf r - p_i|}{c}\right) e^{\,j 2\pi f_{\text{dem}} \tau_i(\mathbf r)}$$

The exponential is the phase compensation from §4. What remains is the choice of
how to reduce the aligned vector `y` to one number, and that is where the design
space is.

### The dynamic aperture

`w_i` is the receive apodization, and it does two jobs.

The **aperture size** is set by holding the f-number `F# = depth / aperture`
constant, so the aperture opens with depth. This keeps the lateral resolution cell
a roughly constant number of wavelengths wide instead of degrading linearly with
depth, and it suppresses the outermost channels, whose echoes are both weak
(element directivity) and geometrically unreliable.

The **window shape** trades mainlobe width against sidelobe level — the same
trade as windowing an FFT, because it is the same mathematics: the beam pattern
is the spatial Fourier transform of the aperture weighting. A rectangular
aperture gives the narrowest mainlobe and −13 dB sidelobes; a Hann window widens
the mainlobe by about 1.5× and buys more than 20 dB of sidelobe suppression.

### Delay and sum

$$y_{\text{DAS}} = \sum_i y_i$$

Linear, data-independent, trivially parallel, and what essentially every clinical
scanner ships. Its resolution and sidelobe level are set entirely by the aperture
and the window — nothing about the data changes them.

### Delay, multiply and sum

$$y_{\text{DMAS}} = \sum_{i<j} \frac{y_i y_j}{\sqrt{|y_i||y_j|}}$$

Multiply every pair, take the signed square root to restore units, and sum. A pair
that agrees in phase reinforces; a pair whose alignment is accidental does not.

The square root restores the *units* that the multiplication squared. It does
**not** halve the phase, and this is worth stating because getting it wrong is an
easy and silent mistake: taking the principal root of the *product*,
`√(y_i y_j)`, halves the phase ramp across the aperture and makes the mainlobe
*wider* than DAS rather than narrower — the exact opposite of the point. (This
engine made that mistake first; `test_combiners` caught it.)

Substituting `v_i = y_i / √|y_i|` gives `ŷ_ij = v_i v_j`, so

$$y_{\text{DMAS}} = \frac{\left(\sum_i v_i\right)^2 - \sum_i v_i^2}{2}$$

which evaluates in one pass instead of the naive O(N²) double loop. The squared
coherent sum is where the gain comes from: the aperture sum is squared, so the
mainlobe narrows and sidelobes fall off twice as fast in dB. Measured on this
engine's point-target test: **0.58 mm versus DAS's 0.99 mm**, a 41 % improvement.

One consequence to know about: the pair product doubles the residual phase, so
the output oscillates axially at twice the rate. On RF this is why the technique
is always followed by a bandpass at 2f₀ ("F-DMAS"); on baseband IQ, DC stays at
DC and no extra filter is needed — but the frame is no longer phase-linear in
displacement, so the quantitative estimators should be run on DAS frames.

### Minimum variance (Capon)

Choose the weights that minimise total output power subject to unity gain on the
look direction:

$$\min_{\mathbf w} \; \mathbf w^H \mathbf R \mathbf w \quad \text{s.t.} \quad \mathbf w^H \mathbf a = 1 \qquad \Longrightarrow \qquad \mathbf w = \frac{\mathbf R^{-1}\mathbf a}{\mathbf a^H \mathbf R^{-1}\mathbf a}$$

Because the data are already delay-compensated, the steering vector `a` is simply
all ones. The optimiser adaptively places nulls on whatever is interfering, which
is why MV resolves so much better than DAS: **0.25 mm versus 0.99 mm** on the
same data, a factor of four.

Two things are not optional extras but the price of making it work at all:

- **Spatial smoothing.** The covariance of a single snapshot is rank 1 and cannot
  be inverted. `R` is estimated by averaging the outer product over overlapping
  subarrays of length `L ≤ N/2` (and, here, over neighbouring range samples).
  This also *decorrelates coherent interference*, which single-snapshot
  estimation cannot do. Shorter subarrays give a better-conditioned estimate and
  worse resolution.
- **Diagonal loading.** Adding `ε·tr(R)/L` to the diagonal stops the adaptive
  weights from nulling the *signal* when the covariance estimate is noisy. It
  continuously trades MV back towards DAS as it increases.

MV costs about 9× DAS in this implementation and is the reason the benchmark
reports it separately.

### Coherence factor

$$\mathrm{CF} = \frac{\left|\sum_i y_i\right|^2}{N \sum_i |y_i|^2}$$

The ratio of coherent to incoherent energy across the aperture: near 1 where the
channels agree (a real target), near 0 where they do not (clutter, off-axis
energy). Applied as a multiplicative weight it costs almost nothing and improves
every contrast metric dramatically — and it is the clearest illustration of why
those metrics need company. See §7.

### Compounding modes

- **Coherent** — sum the aligned channel data across transmits *before*
  combining. Correct for steered plane waves, and what synthesises the transmit
  focus.
- **Incoherent** — beamform each transmit separately and sum magnitudes. Destroys
  phase (so no Doppler downstream), but reduces speckle variance and is robust to
  transmits that are not mutually coherent.
- **Select** — use only the best-illuminating transmit per pixel. This is classic
  line-by-line focused imaging expressed on an arbitrary grid: no blending
  artefacts, phase preserved.

### Transmit sensitivity

A transmit only illuminates part of the field. Outside the region it insonified
there is no transmitted energy, so anything the beamformer finds there is
clutter, and weighting it in makes the image worse. `transmit_sensitivity`
computes the illuminated band by tracing the aperture edges along the propagation
direction (or through the virtual source), widens it by the diffraction-limited
beam width `F#·λ`, and applies a Tukey taper so neighbouring transmits blend
rather than leaving seams.

---

## 6. Resolution, contrast, and the trade between them

**Axial resolution** is set by the pulse length, and the pulse length by the
transducer's fractional bandwidth:

$$\Delta z \approx \frac{c}{2B} \approx \frac{\lambda}{2 \cdot \mathrm{BW}}$$

At 5 MHz with 65 % bandwidth that is about 0.24 mm — and this engine measures
0.27 mm. Axial resolution is a property of the transducer, and no beamformer
improves it.

**Lateral resolution** is set by the aperture:

$$\Delta x \approx F\# \cdot \lambda$$

At F/1.75 and 5 MHz that is 0.54 mm two-way. Lateral resolution is the *only*
one the beamformer controls, which is why the whole beamforming literature is
about the lateral direction. Note the asymmetry: lateral resolution is typically
2–4× worse than axial, which is why lesions in ultrasound images look
horizontally smeared.

`test_resolution_scales_with_aperture` asserts the scaling directly: halving the
f-number must roughly halve the beam width.

---

## 7. Judging a beamformer

Adaptive beamformers are non-linear. They can improve every conventional metric
while making the image *less* useful, and this is not hypothetical — it is what
this engine measures. From `python -m usx compare` on a cyst phantom:

| | lateral FWHM | contrast | gCNR | speckle SNR |
|---|---|---|---|---|
| DAS | 0.81 mm | −34 dB | 0.98 | **1.95** |
| DAS + CF | 0.60 mm | **−93 dB** | 0.98 | **0.89** |
| DMAS | 0.61 mm | −38 dB | 0.88 | 1.10 |
| MV (L=16) | **0.25 mm** | −33 dB | 0.98 | **1.89** |

The coherence factor's −93 dB contrast is not a better image. It is the metric
being gamed: CF suppresses low-coherence pixels, the anechoic cyst is entirely
low-coherence, so it goes to zero and the contrast ratio diverges. The speckle SNR
column tells the truth — it falls from the Rayleigh value of 1.91 to 0.89,
meaning the texture statistics have been destroyed along with the clutter. A
radiologist reads tissue by its texture.

Minimum variance, by contrast, quadruples the resolution while *preserving* the
speckle statistics (1.89 ≈ 1.91). That is a real improvement, and it is visible
in the metrics only because they were reported together.

**gCNR** deserves a note. Defined as one minus the overlap of the two regions'
distributions, it is the probability that a random pixel can be correctly
assigned to lesion or background. Because it depends only on the distributions'
overlap, it is invariant under *any* monotone transform of the pixel values — so
a beamformer that "improves" contrast by stretching the dynamic range moves CNR
and leaves gCNR alone. That invariance is only exact if the *binning* is too;
this engine bins on pooled ranks rather than on values for that reason, and
`test_gcnr_is_invariant_to_monotone_transforms` asserts it.

---

## 8. Doppler

Fire the same look repeatedly at the pulse repetition frequency and watch how the
phase of each pixel evolves. A scatterer moving axially at `v_z` changes the
round-trip path by `2 v_z T` between pulses, so

$$\Delta\phi = -\frac{4\pi f_0 v_z T}{c}$$

The minus sign follows from the IQ convention: increasing depth means a longer
round trip, which the `e^{−j2πft}` mixer turns into a *decreasing* phase.

### Clutter filtering comes first

Blood scatters roughly **40 dB more weakly** than the tissue around it, and that
tissue is not still: vessel walls pulse, the probe moves, the patient breathes.
Estimate velocity without removing the tissue echo first and you measure the wall.
This one step is the difference between a Doppler image and a coloured-in tissue
image.

The engine uses **polynomial regression filtering**: fit and subtract a low-order
polynomial in slow time, per pixel. Tissue motion over a short ensemble is smooth
and well approximated by a low-order trend; blood is not. It works with the very
short ensembles Doppler actually uses (8–32 pulses), where an FIR high-pass
filter's transient would consume the whole packet.

`test_doppler` measures this directly: with static clutter four times stronger
than the flow, the unfiltered estimate returns 1.3 cm/s for a true 15 cm/s, and
the filtered estimate returns 15.1 cm/s.

### Velocity estimation

The classic **Kasai** estimator uses the lag-one slow-time autocorrelation:

$$v_z = -\frac{c\,\mathrm{PRF}}{4\pi f_0}\arg R(0,1)$$

The engine defaults to the **Loupas** 2-D estimator, which additionally uses the
lag-one *fast-time* autocorrelation to estimate the true local centre frequency
rather than assuming `f₀`:

$$\hat f = \frac{c}{4\pi \Delta z}\arg R(1,0), \qquad v_z = -\frac{\Delta z \cdot \mathrm{PRF}}{\arg R(1,0)}\arg R(0,1)$$

This matters because attenuation is frequency-dependent, so the pulse's centre
frequency *drops* with depth; assuming the nominal `f₀` biases velocity low at
depth by several percent. The Loupas form is also elegant in that the assumed
speed of sound cancels out entirely.

### Aliasing

Phase is only unambiguous over ±π, so

$$v_{\text{Nyq}} = \frac{c \cdot \mathrm{PRF}}{4 f_0}$$

Exceed it and the estimate wraps — the familiar colour Doppler artefact where the
centre of a fast jet flips colour. Raising the PRF raises the limit but reduces
the maximum unambiguous depth (`c/2·PRF`), so depth and velocity range trade
directly against each other.

### The angle problem

Doppler measures only the velocity component *along the beam*. Flow exactly
parallel to the transducer face produces no signal at all, which is why
sonographers angle the probe and why reported velocities are divided by
`cos θ`. The engine's `test_lateral_flow_produces_no_axial_doppler` asserts this
explicitly, because it is the most common source of clinically wrong numbers.

### Power Doppler and the coherence gate

Integrating post-filter energy without estimating velocity gives **power
Doppler**: far more sensitive and much less angle-dependent, because it does not
need a resolvable phase slope. It finds slow flow in small vessels, at the cost
of telling you nothing about direction or speed.

Power alone, though, is a weak *discriminator*. What survives the clutter filter
over stationary tissue is receive noise, and because blood is 30–40 dB below
tissue, there can be plenty of it. So the engine also gates on **lag-one
coherence** `|R(0,1)|/R(0,0)`: noise is white in slow time and has near-zero
coherence, while real flow is narrowband and highly correlated pulse to pulse.
This separates the two even when their powers are comparable — and it is the same
quantity as the variance output, since `variance = 2(1 − coherence)`.

---

## 9. Elastography

Palpation, quantified. Compress the tissue slightly, measure how far each point
moved, and differentiate: stiff tissue strains less than soft tissue under the
same stress, so the strain image is a stiffness map.

The displacement estimate uses the same phase-to-distance conversion as Doppler,
without the time axis. Over a small kernel,

$$R = \sum \overline{a}\, b, \qquad \delta = -\frac{c}{4\pi \hat f}\arg R$$

with `f̂` again estimated from the axial phase gradient. The normalized
correlation `|R|/√(E_a E_b)` comes out alongside as a quality measure, and pixels
that decorrelate are masked rather than reported.

Because the estimate is a phase, it unwraps only within ±λ/2 — which is why
elastography uses small compression steps rather than one large one. This engine
measures a 20 µm translation (about λ/15) to within 0.03 µm, and recovers an
applied 0.5 % strain as 0.50 %.

Differentiating a noisy displacement field directly amplifies the noise, so
`axial_strain` fits a line over a window instead — the standard least-squares
strain estimator. The window length is the usual trade: longer is smoother and
blurs the boundary between stiff and soft.

---

## 10. What is not modelled

Every omission below is a place where real ultrasound is harder than this
simulation, listed so that results from the engine are not over-read.

**Linear propagation only.** Real tissue is mildly non-linear, so a finite-
amplitude pulse generates harmonics as it propagates. Clinical scanners exploit
this: **tissue harmonic imaging** receives at 2f₀, where the beam is narrower and
— crucially — near-field reverberation clutter is much weaker, because the
harmonic has not yet built up in the first centimetre. Adding it would mean a
non-linear propagation model (KZK or full-wave), not a change to the beamformer.

**Homogeneous speed of sound.** The beamformer computes delays from an assumed
1540 m/s. Fat is nearer 1450 and muscle nearer 1580, so in a real body the delays
are wrong and the image is defocused — this is **aberration**, and it is the
single largest source of image degradation in clinical ultrasound, especially in
patients with more subcutaneous fat. Modelling it needs a spatially varying speed
map in the simulator; correcting it is an open research area, and estimating the
local speed of sound is one of the quantitative tasks this engine's IQ output is
designed to feed.

**No multiple scattering.** Echoes bounce once. Real reverberation between strong
interfaces (the near-field haze in a clinical image) is absent, which makes
simulated images cleaner than real ones in a specific, recognisable way.

**2-D imaging plane.** Elevation is modelled only as a fixed aperture height. In
reality the elevation slice has finite, depth-varying thickness and everything in
it is projected into one image — **partial volume** effects that make small
structures look larger and lower in contrast than they are.

**Frequency-independent attenuation.** Attenuation is applied at the centre
frequency, so the pulse's amplitude drops with depth but its *spectrum* does not
downshift. The real downshift is a few percent per centimetre; the Loupas
estimator in §8 compensates for it in the velocity estimate, but the simulator
does not produce it.

**Rigid scatterer motion.** Scatterers translate at constant velocity. Real blood
has velocity gradients within a resolution cell (which broadens the Doppler
spectrum — "intrinsic spectral broadening") and the scatterer population
decorrelates as it moves through the beam.

**No electronics model.** No amplifier noise figure, no ADC quantisation, no
element cross-talk, no dead elements. Receive noise is a single additive white
Gaussian term.

**CPU only.** The beamformer is embarrassingly parallel and would map directly
onto a GPU; see [PERFORMANCE.md](PERFORMANCE.md) for what that would buy and
where the remaining serial cost sits.

---

## Further reading

- Szabo, *Diagnostic Ultrasound Imaging: Inside Out* — the standard textbook.
- Jensen, "Field II" — the reference point-scatterer simulator, which computes
  exact spatial impulse responses where this engine approximates them.
- Montaldo et al. (2009), "Coherent plane-wave compounding for very high frame
  rate ultrasonography" — the plane-wave compounding paper.
- Matrone et al. (2015), "The delay multiply and sum beamforming algorithm".
- Synnevåg, Austeng & Holm (2007), "Adaptive beamforming applied to medical
  ultrasound imaging" — minimum variance in this setting.
- Loupas, Powers & Gill (1995), "An axial velocity estimator for ultrasound blood
  flow imaging".
- Rodriguez-Molares et al. (2018), "The generalized contrast-to-noise ratio".
