#!/usr/bin/python3


import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import Optional, Dict, Any
import librosa
import soundfile as sf
import numpy as np
from flax.core import FrozenDict
from flax.serialization import msgpack_serialize, msgpack_restore
from pathlib import Path


class Perceptron(nn.Module):
	input_dim: int
	hidden_dims: list
	output_dim: int
	
	@nn.compact
	def __call__(self, x):
		for n, dim in enumerate(self.hidden_dims):
			x = nn.Dense(dim, name=f'layer_{n}')(x)
			x = nn.relu(x)
		n = len(self.hidden_dims)
		x = nn.Dense(self.output_dim, name=f'layer_{n}')(x)
		return x


class Encoder(nn.Module):
	"Convolutional encoder."
	input_dim: int
	hidden_dims: list
	output_dim: int
	
	kernel: list
	strides: list
	
	@nn.compact
	def __call__(self, x):
		for n, dim in enumerate(self.hidden_dims):
			x = nn.Conv(dim, kernel_size=self.kernel, strides=self.strides, name=f'layer_{n}')(x)
			x = nn.relu(x)
		n = len(self.hidden_dims)
		x = nn.Conv(self.output_dim, kernel_size=self.kernel, strides=self.strides, name=f'layer_{n}')(x)
		return x


class Decoder(nn.Module):
	"Convolutional decoder."
	input_dim: int
	hidden_dims: list
	output_dim: int
	
	kernel: list
	strides: list
	
	@nn.compact
	def __call__(self, x):
		for n, dim in enumerate(self.hidden_dims):
			x = nn.ConvTranspose(dim, kernel_size=self.kernel, strides=self.strides, name=f'layer_{n}')(x)
			x = nn.relu(x)
		n = len(self.hidden_dims)
		x = nn.ConvTranspose(self.output_dim, kernel_size=self.kernel, strides=self.strides, name=f'layer_{n}')(x)
		return x


class Autoencoder(nn.Module):
	encoder: nn.Module
	decoder: nn.Module
	
	def __call__(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


class Trainer:
	def __init__(self):
		self.optimizer = optax.adam(learning_rate=self.learning_rate)
	
	def loss_fn(self, params, batch):
		"Compute the loss for a batch of data."
		predictions = self.autoencoder.apply(params, batch)
		return jnp.mean((batch - predictions) ** 2)
	
	@partial(jax.jit, static_argnames=['self'])
	def update(self, params, opt_state, batch):
		"Update model parameters for a batch of data."
		grads = jax.grad(self.loss_fn)(params, batch)
		updates, opt_state = self.optimizer.update(grads, opt_state)
		params = optax.apply_updates(params, updates)
		return params, opt_state
	
	def save(self, filename):
		"Save weights to file."
		with Path(filename).open('wb') as f:
			f.write(msgpack_serialize(self.params))
	
	def load(self, filename):
		"Load weights from file."
		with Path(filename).open('rb') as f:
			self.params = msgpack_restore(f.read())
		self.verify()
	
	def reset(self, key):
		"Reset weights to default random values."
		self.params = self.autoencoder.init(key, jnp.zeros((1, self.decompressed_dim)))
		self.verify()
	
	def verify(self):
		"Verify if parameters have the right shape."
		for n, (a, b) in enumerate(zip([self.decompressed_dim] + self.hidden_dims, self.hidden_dims + [self.compressed_dim])):
			if self.params['params']['encoder'][f'layer_{n}']['kernel'].shape != (a, b):
				raise ValueError
		
		for n, (a, b) in enumerate(zip([self.decompressed_dim] + list(reversed(self.hidden_dims)), list(reversed(self.hidden_dims)) + [self.decompressed_dim])):
			if self.params['params']['decoder'][f'layer_{n}']['kernel'].shape != (a, b):
				raise ValueError
	
	def train(self, key: jax.random.PRNGKey, data: jnp.ndarray):
		"Train the autoencoder on the given data."
		
		params = self.params
		opt_state = self.optimizer.init(params)
		
		for epoch in range(self.epochs):
			key, epoch_key = jax.random.split(key)
			if self.shuffle:
				shuffled_indices = jax.random.permutation(epoch_key, len(data))
				prepared_data = data[shuffled_indices]
			else:
				prepared_data = data
			
			base = jax.random.randint(epoch_key, shape=(1,), minval=0, maxval=len(data) - 1)[0]
			epoch_loss = 0.0
			for i in range(self.batches_per_epoch):
				batch = prepared_data[base + i * self.batch_size : base + (i + 1) * self.batch_size]
				while len(batch) < self.batch_size:
					batch = jnp.concatenate((batch, prepared_data[:self.batch_size - len(batch)]))
				assert len(batch) == self.batch_size
				params, opt_state = self.update(params, opt_state, batch)
				epoch_loss += self.loss_fn(params, batch)
			
			if epoch % 10 == 0:
				print(f"Compressor trainer, epoch {epoch}/{self.epochs}, loss: {epoch_loss / (len(data) / self.batch_size):.4f}")
		else:
			print(f"Compressor trainer, epoch {self.epochs}/{self.epochs}, loss: {epoch_loss / (len(data) / self.batch_size):.4f}")
		
		self.params = params
	
	def learn(self, key, dataset):
		datas = [[_data, 0, _name] for (_name, _data) in dataset.items()]
		for n in range(len(datas)):
			data = datas[n][0]
			loss = self.loss_fn(self.params, data)
			datas[n][1] = float(loss)
		
		datas.sort(key=lambda _k: _k[1], reverse=True)
		print("initial loss:", [(_d[2], _d[1]) for _d in datas])
		greatest_loss = datas[0][1]
		
		hard_datas = []
		soft_datas = []
		for data, loss, name in datas:
			if loss > greatest_loss / 2:
				hard_datas.append([data, 0, name])
			else:
				soft_datas.append([data, 0, name])
		del datas
		
		for m in range(50):
			for datas in [hard_datas, soft_datas]:
				for n in range(len(datas)):
					data = datas[n][0]
					loss = self.loss_fn(self.params, data)
					datas[n][1] = float(loss)
				
				datas.sort(key=lambda _k: _k[1], reverse=True)
				
				print(m, "loss:", [(_d[2], _d[1]) for _d in datas])
				data = datas[0][0]
				self.train(key, data)
	
	def encode(self, data: jnp.ndarray) -> jnp.ndarray:
		"Encode data using the encoder."
		return self.encoder.apply({'params':self.params['params']['encoder']}, data)
	
	def decode(self, data: jnp.ndarray) -> jnp.ndarray:
		"Decode data using the decoder."
		return self.decoder.apply({'params':self.params['params']['decoder']}, data)


def analyze_audio(file_path: str, sample_rate: int = 22050, chunk_duration_ms: int = 50, n_fft_bins: int = 512) -> jnp.ndarray:
	"""
	Analyze an audio file by splitting it into chunks and computing the Fourier transform for each chunk.
	The FFT results are padded or truncated to a fixed number of frequency bins.
	The output shape is (num_chunks, 2 * n_fft_bins), concatenating real and imaginary parts.

	Args:
		file_path: Path to the audio file (e.g., WAV).
		sample_rate: Target sample rate for the audio (default: 22050 Hz).
		chunk_duration_ms: Duration of each chunk in milliseconds (default: 50 ms).
		n_fft_bins: Fixed number of frequency bins to use (default: 512).

	Returns:
		A 2D JAX NumPy array of shape (num_chunks, 2 * n_fft_bins), where:
		- num_chunks: Number of audio chunks.
		- 2 * n_fft_bins: Concatenated real and imaginary parts of the FFT.
	"""
	# Load audio file
	audio, sr = librosa.load(file_path, sr=sample_rate)
	audio = jnp.array(audio)  # Convert to JAX array

	# Calculate chunk length in samples
	chunk_length = int(sample_rate * chunk_duration_ms / 1000)

	# Pad audio to ensure it's divisible by chunk_length
	pad_length = (chunk_length - (len(audio) % chunk_length)) % chunk_length
	audio_padded = jnp.pad(audio, ((0, pad_length),), mode='constant')

	# Split audio into chunks
	num_chunks = len(audio_padded) // chunk_length
	chunks = [audio_padded[i*chunk_length:(i+1)*chunk_length] for i in range(num_chunks)]
	
	# Compute Fourier transform for each chunk
	fft_results = []
	for chunk in chunks:
		# Compute FFT
		fft = jnp.fft.fft(chunk)
		
		# Take only the first half (real FFT)
		fft_half = fft[:len(fft) // 2]
		
		# Pad or truncate to n_fft_bins
		if len(fft_half) < n_fft_bins:
			# Pad with zeros if too few frequency bins
			pad_width = n_fft_bins - len(fft_half)
			fft_half = jnp.pad(fft_half, (0, pad_width), mode='constant')
		elif len(fft_half) > n_fft_bins:
			# Truncate if too many frequency bins
			fft_half = fft_half[:n_fft_bins]
		
		# Concatenate real and imaginary parts
		fft_real_imag = jnp.concatenate([jnp.real(fft_half), jnp.imag(fft_half)])
		
		fft_results.append(fft_real_imag)
	
	# Stack results into a single array
	return jnp.stack(fft_results)


def synthesize_audio(fft_array: jnp.ndarray, output_path: str, sample_rate: int = 22050, chunk_duration_ms: int = 50, n_fft_bins: int = 512) -> None:
	"""
	Synthesize an audio signal from a Fourier transform array and save it as a WAV file.
	The FFT array is assumed to have shape (num_chunks, 2 * n_fft_bins).

	Args:
		fft_array: 2D JAX NumPy array of shape (num_chunks, 2 * n_fft_bins) containing the FFT data.
		output_path: Path to save the synthesized WAV file.
		sample_rate: Sample rate of the audio (default: 22050 Hz).
		chunk_duration_ms: Duration of each chunk in milliseconds (default: 50 ms).
		n_fft_bins: Fixed number of frequency bins used (default: 512).
	"""
	# Calculate chunk length in samples
	chunk_length = int(sample_rate * chunk_duration_ms / 1000)

	# Synthesize each chunk from FFT
	reconstructed_chunks = []
	for chunk_fft in fft_array:
		# Split into real and imaginary parts
		real_part = chunk_fft[:n_fft_bins]
		imag_part = chunk_fft[n_fft_bins:2*n_fft_bins]

		# Create full FFT array (symmetric for real signals)
		full_fft = jnp.zeros(chunk_length, dtype=jnp.complex64)
		full_fft = full_fft.at[:n_fft_bins].set(real_part + 1j * imag_part)

		# Mirror the positive frequencies to create the full FFT
		if n_fft_bins > 1:
			mirrored_part = jnp.conj(real_part[1:n_fft_bins] + 1j * imag_part[1:n_fft_bins])[::-1]
			full_fft = full_fft.at[-n_fft_bins+1:].set(mirrored_part)

		# Inverse FFT to get the time-domain signal
		reconstructed_chunk = jnp.fft.ifft(full_fft).real

		reconstructed_chunks.append(reconstructed_chunk)

	# Combine chunks into a single audio signal
	reconstructed_audio = jnp.concatenate(reconstructed_chunks)

	# Convert to regular NumPy array for saving with soundfile
	sf.write(output_path, np.asarray(reconstructed_audio), sample_rate)



class CompressorTrainer(Trainer):	
	def __init__(self):
		self.decompressed_dim = 1024
		self.hidden_dims = [1200]
		self.compressed_dim = 80
		
		self.learning_rate = 0.00005
		self.epochs = 10
		self.batches_per_epoch = 10
		self.batch_size = 50
		self.shuffle = True
		
		self.encoder = Perceptron(input_dim=self.decompressed_dim, hidden_dims=self.hidden_dims, output_dim=self.compressed_dim)
		self.decoder = Perceptron(input_dim=self.compressed_dim, hidden_dims=list(reversed(self.hidden_dims)), output_dim=self.decompressed_dim)
		self.autoencoder = Autoencoder(encoder=self.encoder, decoder=self.decoder)
		super().__init__()
	
	def reset(self, key):
		self.params = self.autoencoder.init(key, jnp.zeros((1, self.decompressed_dim)))
		self.verify()
	
	def verify(self):
		for n, (a, b) in enumerate(zip([self.decompressed_dim] + self.hidden_dims, self.hidden_dims + [self.compressed_dim])):
			if self.params['params']['encoder'][f'layer_{n}']['kernel'].shape != (a, b):
				raise ValueError(f"Encoder layer {n}: {self.params['params']['encoder'][f'layer_{n}']['kernel'].shape} != {(a, b):}")
		
		for n, (a, b) in enumerate(zip([self.compressed_dim] + list(reversed(self.hidden_dims)), list(reversed(self.hidden_dims)) + [self.decompressed_dim])):
			if self.params['params']['decoder'][f'layer_{n}']['kernel'].shape != (a, b):
				raise ValueError(f"Decoder layer {n}: {self.params['params']['decoder'][f'layer_{n}']['kernel'].shape} != {(a, b):}")


class ConvolutionalTrainer(Trainer):
	def __init__(self):
		self.decompressed_dim = 20
		self.compressed_dim = 10
		self.hidden_dims = [80, 40]
		self.kernel = [5]
		self.strides = [2]
		
		self.learning_rate = 0.0000075
		self.epochs = 100
		self.batches_per_epoch = 5
		self.batch_size = 200
		self.shuffle = False
		
		self.encoder = Encoder(input_dim=self.decompressed_dim, hidden_dims=self.hidden_dims, output_dim=self.compressed_dim, kernel=self.kernel, strides=self.strides)
		self.decoder = Decoder(input_dim=self.compressed_dim, hidden_dims=list(reversed(self.hidden_dims)), output_dim=self.decompressed_dim, kernel=self.kernel, strides=self.strides)
		self.autoencoder = Autoencoder(encoder=self.encoder, decoder=self.decoder)
		super().__init__()
	
	def reset(self, key: jax.random.PRNGKey):
		self.params = self.autoencoder.init(key, jnp.zeros((1, 1, self.decompressed_dim)))
		self.verify()
	
	def verify(self):
		for n, (a, b) in enumerate(zip([self.decompressed_dim] + self.hidden_dims, self.hidden_dims + [self.compressed_dim])):
			if self.params['params']['encoder'][f'layer_{n}']['kernel'].shape != tuple(self.kernel) + (a, b):
				raise ValueError
		
		for n, (a, b) in enumerate(zip([self.compressed_dim] + list(reversed(self.hidden_dims)), list(reversed(self.hidden_dims)) + [self.decompressed_dim])):
			if self.params['params']['decoder'][f'layer_{n}']['kernel'].shape != tuple(self.kernel) + (a, b):
				raise ValueError


if __name__ == '__main__':
	from time import time
	key = jax.random.PRNGKey(int(time() * 1e9))
	
	compressor_trainer = CompressorTrainer()
	convolutional_trainer = ConvolutionalTrainer()
	
	key, subkey = jax.random.split(key)
	data_paths = list(Path('training').iterdir())
	training_paths = []
	for i in jax.random.choice(subkey, len(data_paths), shape=(len(data_paths) // 4,), replace=False):
		training_paths.append(data_paths[i])
	print(" ".join(_p.name for _p in training_paths))
	
	training_set = {}
	for path in training_paths:
		audio = analyze_audio(path, chunk_duration_ms=50, n_fft_bins=512)
		training_set[path.name] = audio
	#training_data = jnp.concatenate(training_set)
	
	key, subkey = jax.random.split(key)
	compressor_trainer.reset(subkey)	
	key, subkey = jax.random.split(key)
	convolutional_trainer.reset(subkey)
	
	key, subkey = jax.random.split(key)
	try:
		compressor_trainer.load('compressor.bin')
	except IOError:
		compressor_trainer.learn(subkey, training_set)
		compressor_trainer.save('compressor.bin')
	
	compressor_dir = Path('result_compressor')
	compressor_dir.mkdir(exist_ok=True)
	for path in data_paths:
		audio = analyze_audio(path, chunk_duration_ms=50, n_fft_bins=512)
		chunks = compressor_trainer.encode(audio)
		print(path.name, audio.shape, chunks.shape)
		audio = compressor_trainer.decode(chunks)
		synthesize_audio(audio, compressor_dir / path.name, chunk_duration_ms=50, n_fft_bins=512)
	
	quit()
	#chunks = compressor_trainer.encode(training_data)

	key, subkey = jax.random.split(key)
	try:
		convolutional_trainer.load('convolution.bin')
	except IOError:
		convolutional_trainer.train(subkey, chunks)
		convolutional_trainer.save('convolution.bin')
	
	features = convolutional_trainer.encode(chunks)
	
	#print(audio.shape, chunks.shape, features.shape)
	
	convolutional_dir = Path('result_convolutional')
	convolutional_dir.mkdir(exist_ok=True)	
	for path in data_paths:
		audio = analyze_audio(path, chunk_duration_ms=50, n_fft_bins=512)
		chunks = compressor_trainer.encode(audio)
		features = convolutional_trainer.encode(chunks)
		print(path.name, audio.shape, chunks.shape, features.shape)
		chunks = convolutional_trainer.decode(features)
		audio = compressor_trainer.decode(chunks)
		synthesize_audio(audio, convolutional_dir / path.name, chunk_duration_ms=50, n_fft_bins=512)


