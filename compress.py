#!/usr/bin/python3


from typing import Any, Tuple, Union, Optional, Self
from collections.abc import Generator, Container

import sys
from itertools import tee
from pathlib import Path

import jax
import jax.numpy as np

import soundfile as sf


class NDArrayMeta(type):
	def __instancecheck__(self, value: Any) -> bool:
		if not hasattr(self, 'shape'):
			raise ValueError
		
		if not hasattr(value, 'shape'):
			return False
		
		if not all(isinstance(_dim, int) for _dim in value.shape):
			return False
		
		if any(_dim < 0 for _dim in value.shape):
			return False
		
		if Ellipsis not in self.shape:
			if len(value.shape) != len(self.shape):
				return False
		else:
			if len(value.shape) < len(self.shape) - 1:
				return False
		
		for v, s in zip(value.shape, self.shape):
			if s is Ellipsis:
				break
			elif isinstance(s, int):
				if v != s:
					return False
			elif isinstance(s, (range, Container)):
				if v not in s:
					return False
			elif isinstance(s, slice):
				if s.start is not None and v < s.start:
					return False
				if s.stop is not None and v >= s.stop:
					return False
				if s.step is not None and (v - (s.start or 0)) % s.step != 0:
					return False
			else:
				raise ValueError(f"Unsupported shape element: {s}")
		
		if Ellipsis in self.shape:
			for v, s in zip(reversed(value.shape), reversed(self.shape)):
				if s is Ellipsis:
					break
				elif isinstance(s, int):
					if v != s:
						return False
				elif isinstance(s, (range, Container)):
					if v not in s:
						return False
				elif isinstance(s, slice):
					if s.start is not None and v < s.start:
						return False
					if s.stop is not None and v >= s.stop:
						return False
					if s.step is not None and (v - (s.start or 0)) % s.step != 0:
						return False
				else:
					raise ValueError(f"Unsupported shape element: {s}")
		
		return True


class NDArray(metaclass=NDArrayMeta):
	"""
	An abstract class representing a numeric array. In its default state it is an empty type (doesn't have any value).
	When it receives a generic argument it becomes nonempty.
	Providing `None` makes it represent a scalar.
	Providing a single integer, a single slice, a one-tuple with integer or one-tuple with a slice makes it represent a vector.
	Providing a 2-tuple makes it a matrix. Providing 3-tuple makes it a rank 3 tensor and so on.
	Providing an Ellipsis (...) makes it represent any of the above types.
	Each element of the argument tuple may be an integer, a slice or a container supporting membership test.
	"""
	def __class_getitem__(cls, shape: Union[None, Ellipsis, Tuple[Union[int, slice, range, Container], ...]]):
		if not __debug__:
			return cls

		if shape is None:
			shape = ()
		elif isinstance(shape, tuple):
			if not shape:
				shape = None
		else:
			shape = (shape,)
		
		assert isinstance(shape, tuple)
		
		if not all(isinstance(_arg, (int, slice, range, Container)) or _arg is Ellipsis for _arg in shape):
			raise ValueError("Shape elements must be int, slice, range, Container or Ellipsis.")
		
		return type(cls.__name__, (cls,), {'shape': shape})
	
	def __new__(cls, value: Any) -> Self:
		"""
		Take an ordinary Python value (float, list of floats, list of list of floats etc.).
		Return a jax array with that value. Check type. If the value is a jax array, return it unmodified but check type.
		"""
		result = np.asarray(value) if not isinstance(value, jax.Array) else value
		
		if __debug__ and not isinstance(result, cls):
			raise TypeError(f"Value {value} of shape {value.shape} does not match shape {cls.shape}")

		return result


default_sample_rate = 22050 # 22050 Hz audio sample rate
default_frame_duration = 20 # 50ms frame, 50 frames per second
default_frequencies = 1024 # 1024 frequencies in Fourier transform



def interpolate(frame: NDArray[:], in_sample_rate: int, out_sample_rate: int) -> NDArray[:]:
	"""
	Resample a frame (1D array) by the given ratio using linear interpolation.
	Args:
		frame: Input frame as a 1D array.
		Resampling ratio = out_sample_rate / in_sample_rate
	Returns:
		Resampled frame as a 1D array.
	"""
	#print(len(frame), in_sample_rate, out_sample_rate, len(frame) * out_sample_rate // in_sample_rate)
	x_old = np.linspace(0, 1, len(frame))
	x_new = np.linspace(0, 1, len(frame) * out_sample_rate // in_sample_rate)
	return NDArray[:](np.interp(x_new, x_old, NDArray[:](frame)))


def resample(data: Generator[NDArray[:], None, None], in_sample_rate: int, out_sample_rate: int) -> Generator[NDArray[:], None, None]:
	"""
	Resample frames from in_sample_rate to out_sample_rate. The frames have the same duration.
	Args:
		data: Generator yielding frames (1D arrays).
		in_sample_rate: Input sample rate.
		out_sample_rate: Output sample rate.
	Yields:
		Resampled frames as 1D arrays.
	"""
	for frame in data:
		frame = NDArray[:](frame)
		yield NDArray[:](interpolate(frame, in_sample_rate, out_sample_rate))


def reframe(data: Generator[NDArray[:], None, None], sample_rate: int = default_sample_rate, frame_duration: int = default_frame_duration) -> Generator[NDArray[:], None, None]:
	"""
	Receive frames of any duration, with the specified sample rate. Emit frames of desired duration (in milliseconds).
	Args:
		data: Generator yielding frames (1D arrays).
		sample_rate: Sample rate in Hz.
		frame_duration: Desired frame duration in milliseconds.
	Yields:
		Frames of the desired duration as 1D arrays.
	"""
	samples_per_frame = sample_rate * frame_duration // 1000
	buffer = []
	buffer_length = 0
	for in_frame in data:
		in_frame = NDArray[:](in_frame)
		buffer.append(in_frame)
		buffer_length += len(in_frame)
		
		while buffer_length >= samples_per_frame:
			if len(buffer) > 1:
				b = np.concatenate(buffer)
			else:
				b = buffer[0]
			for n in range(buffer_length // samples_per_frame):
				out_frame = NDArray[samples_per_frame](b[n * samples_per_frame : (n + 1) * samples_per_frame])
				yield out_frame
			buffer = [b[-(buffer_length % samples_per_frame):]]
			buffer_length %= samples_per_frame
	
	if buffer:
		yield NDArray[:](np.concatenate(buffer))


def accumulate(data: Generator[NDArray[...], None, None], sample_size: Optional[int] = None, skip: int = 1) -> NDArray[:, ...]:
	def filter_data(data, sample_size, skip):
		for n, frame in enumerate(data):
			if sample_size is not None and n > sample_size * skip:
				break
			if n % skip == 0:
				yield frame
	
	return NDArray[:, ...](np.stack(list(filter_data(data, sample_size, skip))))


def load_audio_files(dirname: Path, sample_rate: int = default_sample_rate, frame_duration: int = default_frame_duration) -> Generator[NDArray[:], None, None]:
	"""
	Load all WAV files in a directory, resample and reframe them.
	Args:
		dirname: Directory containing WAV files.
		sample_rate: Desired output sample rate.
		frame_duration: Desired frame duration in milliseconds.
	Yields:
		Frames as 1D arrays.
	"""
	for filename in dirname.iterdir():
		print("load", filename)
		yield from load_audio_file(filename, sample_rate, frame_duration)


def load_audio_file(filename: Path, sample_rate: int = default_sample_rate, frame_duration: int = default_frame_duration) -> Generator[NDArray[:], None, None]:
	"""
	Load a WAV file, resample and reframe it.
	Args:
		filename: Path to the WAV file.
		sample_rate: Desired output sample rate.
		frame_duration: Desired frame duration in milliseconds.
	Yields:
		Frames as 1D arrays.
	"""
	with sf.SoundFile(filename, mode='r') as soundfile:
		samples_per_frame = soundfile.samplerate * frame_duration // 1000
		#print(soundfile.samplerate, sample_rate, samples_per_frame)
		while True:
			raw_data = soundfile.read(frames=samples_per_frame, dtype='float32')
			if len(raw_data) == 0:
				break
			mono_data = np.mean(raw_data, axis=1) if raw_data.ndim > 1 else raw_data
			if soundfile.samplerate == sample_rate:
				yield NDArray[:](mono_data)
			else:
				yield from resample([NDArray[:](mono_data)], soundfile.samplerate, sample_rate)


def save_audio_file(filename: Path, data: Generator[NDArray[:], None, None], sample_rate: int = default_sample_rate, volume: float = 1.0) -> None:
	"""
	Save frames to a WAV file.
	Args:
		filename: Output WAV file path.
		data: Generator yielding frames (1D arrays).
		sample_rate: Sample rate in Hz.
	"""
	with sf.SoundFile(filename, mode='w', samplerate=sample_rate, channels=1) as soundfile:
		for frame in data:
			chunk = NDArray[:](frame).reshape(-1, 1)
			soundfile.write(chunk * volume)


def fourier(data: Generator[NDArray[:], None, None], frequencies: int = default_frequencies) -> Generator[NDArray[:], None, None]:
	"""
	Compute the Fourier transform for each frame.
	Args:
		data: Generator yielding frames (1D arrays).
		frequencies: Number of frequency bins.
	Yields:
		Fourier coefficients as 1D arrays (real and imaginary parts concatenated).
	"""
	for window in data:
		window = NDArray[:](window)
		freqs = np.fft.rfft(window)
		freqs = freqs.real[:frequencies]
		if len(freqs) < frequencies:
			freqs = np.concatenate([freqs, np.zeros(frequencies - len(freqs), dtype='float32')])
		yield NDArray[frequencies](freqs)


def inverse_fourier(data: Generator[NDArray[:], None, None], sample_rate: int = default_sample_rate, frame_duration: int = default_frame_duration) -> Generator[NDArray[:], None, None]:
	"""
	Compute the inverse Fourier transform for each frame.
	Args:
		data: Generator yielding Fourier coefficients (1D arrays, real and imaginary parts concatenated).
		sample_rate: Sample rate in Hz.
		frame_duration: Duration of each frame in milliseconds.
	Yields:
		Time-domain frames as 1D arrays of the original length.
	"""
	samples_per_frame = sample_rate * frame_duration // 1000
	for freqs in data:
		freqs = NDArray[:](freqs)
		
		# Inverse FFT with the original length
		if len(freqs) < samples_per_frame // 2 + 2:
			freqs = np.concatenate([freqs, np.zeros(samples_per_frame // 2 + 2 - len(freqs), dtype='float32')])
		
		window = np.fft.irfft(freqs, n=samples_per_frame)
		yield NDArray[samples_per_frame](window)


def crosscorrelation(data: Generator[NDArray[:], None, None], sample_size: Optional[int] = None, skip: int = 1) -> NDArray[:, :]:
	"""
	Compute the cross-correlation matrix for a series of frames.
	Args:
		data: Generator yielding frames (1D arrays).
		sample_size: Maximum number of frames to process.
		skip: Process only 1 / skip frames.
	Returns:
		Cross-correlation matrix as a 2D array.
	"""
	samples = []
	for i, vec in enumerate(data):
		if sample_size is not None and i >= sample_size * skip:
			break
		if i % skip == 0:
			samples.append(NDArray[:](vec))

	if not samples:
		return NDArray[:, :](np.zeros((0, 0)))

	samples = np.array(samples)
	corr_matrix = np.corrcoef(samples.T)
	corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
	corr_matrix = np.fill_diagonal(corr_matrix, 1.0, inplace=False)

	return NDArray[:, :](corr_matrix)


def crosscovariance(data: Generator[NDArray[:], None, None], sample_size: Optional[int] = None, skip: int = 1) -> NDArray[:, :]:
	sample = NDArray[:, :](accumulate(data, sample_size, skip))
	
	corr_matrix = np.corrcoef(samples.T)
	corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
	corr_matrix = np.fill_diagonal(corr_matrix, 1.0, inplace=False)

	return NDArray[:, :](corr_matrix)


def mean_variance(data: Generator[NDArray[:], None, None]) -> Tuple[NDArray[:], NDArray[:]]:
	"""
	Compute the mean and variance of each component independently from a generator of vectors.
	
	Args:
		data: Generator yielding vectors (1D arrays).

	Returns:
		Tuple of (mean, variance) for each component as 1D arrays.
	"""
	# Initialize variables to accumulate sums and squares
	sum_vec = None
	sum_sq_vec = None
	count = 0
	
	for vec in data:
		vec = NDArray[:](vec)
		count += 1
		
		if sum_vec is None:
			sum_vec = np.zeros_like(vec, dtype=float)
			sum_sq_vec = np.zeros_like(vec, dtype=float)
		
		# Accumulate sums and sums of squares
		sum_vec += vec
		sum_sq_vec += np.square(vec)
	
	if count == 0:
		# Return zeros if no data was provided
		return NDArray[:](np.zeros(0)), NDArray[:](np.zeros(0))
	
	# Compute mean and variance
	mean = sum_vec / count
	variance = (sum_sq_vec / count) - np.square(mean)
	
	return NDArray[:](mean), NDArray[:](variance)


def normalize(data:Generator[NDArray[:], None, None], mean:NDArray[:], variance:NDArray[:], threshold: float = 0.001) -> Generator[NDArray[:], None, None]:
	"Normalize the stream of vectors, taking the mean of each element to 0 and variance to 1 or 0, given the provided mean and variance arguments. If abs(v) < threshold the variance is assumed to be 0."
	
	if __debug__:
		mean = NDArray[:](mean)
		variance = NDArray[:](variance)
		if len(mean.shape) != 1:
			raise TypeError
		if mean.shape != variance.shape:
			raise TypeError
	
	for frame in data:
		frame = NDArray[mean.shape[0]](frame)
		normalized = (frame - mean) / np.sqrt(variance)
		result = np.where(np.abs(variance) < threshold, 0.0, normalized)
		yield NDArray[mean.shape[0]](result)


def denormalize(data:Generator[NDArray[:], None, None], mean:NDArray[:], variance:NDArray[:], threshold: float = 0.001) -> Generator[NDArray[:], None, None]:
	"Recover original stream from normalized stream, restoring the original mean and variance, given the provided mean and variance arguments. If abs(v) < threshold the variance is assumed to be 0."
	
	if __debug__:
		mean = NDArray[:](mean)
		variance = NDArray[:](variance)
		if len(mean.shape) != 1:
			raise TypeError
		if mean.shape != variance.shape:
			raise TypeError
	
	mean = NDArray[:](mean)
	variance = NDArray[:](variance)
	for frame in data:
		frame = NDArray[mean.shape[0]](frame)
		yield NDArray[mean.shape[0]](frame * np.where(np.abs(variance) < threshold, 0.0, np.sqrt(variance)) + mean)


def compression_matrix(cross_corr_matrix: NDArray[:, :], n_components: int) -> NDArray[:, :]:
	"""
	Construct a PCA-based compression matrix from a cross-correlation matrix.

	Args:
		cross_corr_matrix: Cross-correlation matrix (square, symmetric).
		n_components: Number of principal components to retain.

	Returns:
		Compression matrix (rectangular, shape: [input_dim, n_components]).
	"""
	
	cross_corr_matrix = NDArray[:, :](cross_corr_matrix)
	
	dim = cross_corr_matrix.shape[0]
	assert cross_corr_matrix.shape == (dim, dim)
	
	# Eigendecomposition
	eigenvalues, eigenvectors = np.linalg.eig(cross_corr_matrix)
	eigenvalues = eigenvalues.real
	eigenvectors = eigenvectors.real
	eigenvectors /= np.linalg.norm(eigenvectors, axis=0)
	
	# Sort eigenvalues and eigenvectors by absolute value
	idx = np.argsort(np.abs(eigenvalues))[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:, idx]
	
	#eigenvectors_inv = np.linalg.inv(eigenvectors)
	
	# Select top N components
	top_eigenvectors = eigenvectors[:, :n_components]
	#top_eigenvectors_inv = eigenvectors_inv[:n_components, :]
	
	return NDArray[dim, n_components](top_eigenvectors)


def compress(data: Generator[NDArray[:], None, None], compression_matrix: NDArray[:, :]) -> Generator[NDArray[:], None, None]:
	"""
	Compress a stream of vectors using the compression matrix.

	Args:
		data: Generator yielding input vectors.
		compression_matrix: PCA-based compression matrix.

	Yields:
		Compressed vectors (shorter vectors).
	"""
	compression_matrix = NDArray[:, :](compression_matrix)
	decompressed_dim, compressed_dim = compression_matrix.shape
	for vec in data:
		vec = NDArray[decompressed_dim](vec)
		compressed = np.dot(vec, compression_matrix)
		yield NDArray[compressed_dim](compressed)


def decompress(data: Generator[NDArray[:], None, None], compression_matrix: NDArray[:, :]) -> Generator[NDArray[:], None, None]:
	"""
	Decompress a stream of compressed vectors using decompression matrix.

	Args:
		data: Generator yielding compressed vectors.
		compression_matrix: PCA-based compression matrix.

	Yields:
		Decompressed vectors (original dimension).
	"""
	compression_matrix = NDArray[:, :](compression_matrix)
	decompressed_dim, compressed_dim = compression_matrix.shape
	for vec in data:
		vec = NDArray[compressed_dim](vec)
		decompressed = np.dot(vec, compression_matrix.T)
		yield NDArray[decompressed_dim](decompressed)


def autocorrelation(data: Generator[NDArray[None], None, None], maxdelay: int, sample_size: Optional[int] = None) -> NDArray[:, :]:
	"""
	Compute the autocorrelation matrix for a series of scalars.
	Args:
		data: Generator yielding scalars.
		maxdelay: Maximum delay for autocorrelation.
		sample_size: Maximum number of samples to process.
	Returns:
		Autocorrelation matrix as a 2D array.
	"""
	samples = []
	for i, sample in enumerate(data):
		if sample_size is not None and i >= sample_size:
			break
		samples.append(NDArray[None](sample))
	if not samples:
		return NDArray[:, :](np.zeros((maxdelay, maxdelay)))
	samples = np.array(samples)
	acorr = np.array([np.corrcoef(samples[:-i], samples[i:])[0, 1] for i in range(1, maxdelay + 1)])
	return NDArray[:, :](np.diag(acorr))


def eigendecomposition(matrix: NDArray[:, :]) -> Tuple[NDArray[:], NDArray[:, :]]:
	"""
	Compute the eigendecomposition of a square matrix.
	Args:
		matrix: Square matrix as a 2D array.
	Returns:
		Tuple of eigenvalues (1D array) and eigenvectors (2D array).
	"""
	matrix = NDArray[:, :](matrix)
	eigenvalues, eigenvectors = np.linalg.eig(matrix)
	return NDArray[:](eigenvalues.real), NDArray[:, :](eigenvectors.real)


if False and __debug__ and __name__ == '__main__':
	audio = load_audio_file(Path('angry/training/2L6hzfX0Ipk.wav'), frame_duration=80)
	for n in range(3):
		assert next(audio).shape == (default_sample_rate * 80 // 1000,)
	audio = reframe(audio)
	for n in range(3):
		assert next(audio).shape == (default_sample_rate * default_frame_duration // 1000,)
	freqs = fourier(audio)
	assert next(freqs).shape == (default_frequencies,)
	freqs = accumulate(freqs, skip=2)
	assert not np.isnan(freqs).any()
	m, v = mean_variance(freqs)
	assert not np.isnan(m).any()
	assert not np.isnan(v).any()
	
	#print(freqs[0][0], m[0], v[0], (freqs[0][0] - m[0]) / np.sqrt(v[0]))
	freqs_norm = normalize(freqs, m, v)
	
	freqs_norm1, freqs_norm2, freqs_norm3 = tee(freqs_norm, 3)
	assert not np.isnan(accumulate(freqs_norm3)).any()
	
	mn, vn = mean_variance(freqs_norm1)
	assert all(abs(_m) < 10**-6 for _m in mn), "All mean values of normalized stream should be close to 0."
	assert all(abs(_v - 1) < 10**-4 or _v == 0 for _v in vn), "All variance values of normalized stream should be close to 1 or should equal 0."
	
	freqs_denorm = denormalize(freqs_norm2, m, v)
	freqs_denorm = accumulate(freqs_denorm)
	#print(freqs_denorm.shape)
	
	assert freqs_denorm.shape == freqs.shape
	#print(freqs[0][0], (freqs[0][0] - m[0]) / np.sqrt(v[0]), freqs_denorm[0][0])
	#print(freqs_denorm[0][0])
	
	#print(freqs - freqs_denorm)
	assert (np.abs(freqs - freqs_denorm) < 10**-5).all()



#quit()


def load_normalized_audio_files(dirname):
	for filename in dirname.iterdir():
		print("load", filename)
		audio = load_audio_file(filename)
		freqs = fourier(audio)
		freqs1, freqs2 = tee(freqs)
		mean, variance = mean_variance(freqs1)
		yield from normalize(freqs2, mean, variance)


if __name__ == '__main__':
	try:
		dirname = Path(sys.argv[1])
	except IndexError:
		print("Usage: {sys.argv[0]} <directory name with 'training' subdir>")
		exit(1)
	
	training_dir = dirname / 'training'
	result_dir = dirname / 'result'
	
	try:
		compression = np.load(dirname / 'compression_matrix.npy')
	except IOError:
		audio = load_normalized_audio_files(training_dir)
		cross_correlation = crosscorrelation(audio)
		compression = compression_matrix(cross_correlation, 80)
		np.save(dirname / 'compression_matrix.npy', compression)
	
	#print(compression)
	#print(decompression)
	
	audio = load_audio_file(list(training_dir.iterdir())[0])
	freqs = fourier(audio)
	mean, variance = mean_variance(freqs)
	
	print()
	result_dir.mkdir(exist_ok=True)
	for filename in training_dir.iterdir():
		print("process", filename)
		audio = load_audio_file(filename)
		freqs = fourier(audio)
		#freqs1, freqs2 = tee(freqs)
		#mean, variance = mean_variance(freqs1)
		normalized_freqs = normalize(freqs, mean, variance)
		compressed_freqs = compress(normalized_freqs, compression)
		normalized_freqs = decompress(compressed_freqs, compression)
		freqs = denormalize(normalized_freqs, mean, variance)
		audio = inverse_fourier(freqs)
		save_audio_file(result_dir / filename.name, audio, volume=5)



