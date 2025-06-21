const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const PopulationSize = 80;
const storagePath = path.join(__dirname, 'storage');
const datasetPath = path.join(__dirname, 'dataset');

const populationConfigs = {
	Generator: {
		layers: 7,
		channels: 4,
		bitsPerRule: 512,
		genotypeSize: 512 * 7 * 4,
		mutation: 20,
		mutategen: 32
	},
	Discriminator: {
		layers: 6,
		channels: 4,
		bitsPerRule: 512,
		genotypeSize: 512 * 6 * 4,
		mutation: 25,
		mutategen: 16
	}
};

let populations = {};

function ensureStorageDir() {
	if (!fs.existsSync(storagePath)) {
		fs.mkdirSync(storagePath);
	}
	if (!fs.existsSync(datasetPath)) {
		fs.mkdirSync(datasetPath);
	}
}

function getPopulationFile(type, key) {
	return path.join(storagePath, `${type.toLowerCase()}_${key}.json`);
}

function getFitnessHistoryFile() {
	return path.join(storagePath, 'fitness_history.json');
}

function saveFitnessHistory(fitnessHistory) {
	ensureStorageDir();
	fs.writeFileSync(getFitnessHistoryFile(), JSON.stringify(fitnessHistory, null, 2));
}

function loadFitnessHistory() {
	ensureStorageDir();
	const historyPath = getFitnessHistoryFile();
	if (fs.existsSync(historyPath)) {
		return JSON.parse(fs.readFileSync(historyPath, 'utf8'));
	}
	return [];
}

function savePopulation(type) {
	ensureStorageDir();
	const pop = populations[type];
	fs.writeFileSync(getPopulationFile(type, "population"), JSON.stringify(pop.population));
	fs.writeFileSync(getPopulationFile(type, "fitness"), JSON.stringify(pop.fitness));
	return true;
}

function resumePopulation(type) {
	ensureStorageDir();
	const popPath = getPopulationFile(type, "population");
	const fitPath = getPopulationFile(type, "fitness");

	if (!fs.existsSync(popPath) || !fs.existsSync(fitPath)) {
		return false;
	}

	const config = populationConfigs[type];
	populations[type] = {
		population: JSON.parse(fs.readFileSync(popPath, 'utf8')),
		fitness: JSON.parse(fs.readFileSync(fitPath, 'utf8')),
		genotypeSize: config.genotypeSize,
		mutation: config.mutation,
		mutategen: config.mutategen
	};

	return true;
}

function newPopulation(type) {
	const config = populationConfigs[type];
	populations[type] = {
		population: [],
		fitness: [],
		genotypeSize: config.genotypeSize,
		mutation: config.mutation,
		mutategen: config.mutategen
	};

	const p = populations[type];

	for (let n = 0; n < PopulationSize; n++) {
		p.population[n] = [];
		p.fitness[n] = 0;
		for (let i = 0; i < p.genotypeSize; i++) {
			p.population[n][i] = Math.round(Math.random());
		}
	}

	savePopulation(type);
}

function clearFitness(type) {
	const p = populations[type];
	for (let n = 0; n < PopulationSize; n++) {
		p.fitness[n] = 0;
	}
	savePopulation(type);
}

function initPopulation(type) {
	if (!resumePopulation(type)) {
		newPopulation(type);
	}
}

function sortf(a, b) {
	return b[1] - a[1];
}

function shufflePopulation(p) {
	for (let i = p.population.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1));
		[p.population[i], p.population[j]] = [p.population[j], p.population[i]];
		[p.fitness[i], p.fitness[j]] = [p.fitness[j], p.fitness[i]];
	}
}

function evolute(type) {
	const p = populations[type];
	const sizehalf = PopulationSize / 2;
	const sizequarter = sizehalf / 2;

	let arrayt = [];
	for (let n = 0; n < PopulationSize; n++) {
		arrayt[n] = [p.population[n], p.fitness[n], n];
	}

	arrayt.sort(sortf);
	arrayt.length = sizehalf;

	p.population = [];
	p.fitness = [];

	for (let i = 0; i < sizequarter; i++) {
		const i0 = i * 4;
		const i1 = i * 4 + 1;
		const i2 = i * 4 + 2;
		const i3 = i * 4 + 3;

		const parent1 = arrayt.splice(Math.floor(Math.random() * arrayt.length), 1)[0][0];
		const parent2 = arrayt.splice(Math.floor(Math.random() * arrayt.length), 1)[0][0];

		const child1 = [];
		const child2 = [];

		for (let j = 0; j < p.genotypeSize; j++) {
			const gen = Math.round(Math.random());
			child1[j] = gen === 1 ? parent1[j] : parent2[j];
			child2[j] = gen === 1 ? parent2[j] : parent1[j];
		}

		p.population[i0] = parent1;
		p.population[i1] = parent2;
		p.population[i2] = child1;
		p.population[i3] = child2;

		p.fitness[i0] = 0;
		p.fitness[i1] = 0;
		p.fitness[i2] = 0;
		p.fitness[i3] = 0;
	}

	const m = 100 / p.mutation;
	for (let i = 0; i < PopulationSize; i++) {
		if (Math.floor(Math.random() * m) === 0) {
			const flips = Math.floor(Math.random() * p.mutategen) + 1;
			for (let j = 0; j < flips; j++) {
				const gen = Math.floor(Math.random() * p.genotypeSize);
				p.population[i][gen] ^= 1;
			}
		}
	}
	
	shufflePopulation(p);
	savePopulation(type);
}

function recreatePopulation(type) {
	newPopulation(type);
}


function unflattenRules(flat, type) {
	const config = populationConfigs[type];
	const { layers, channels, bitsPerRule } = config;

	let rulesArray = [];
	let idx = 0;
	for (let i = 0; i < layers; i++) {
		rulesArray[i] = [];
		for (let j = 0; j < channels; j++) {
			rulesArray[i][j] = [];
			for (let k = 0; k < bitsPerRule; k++) {
				rulesArray[i][j][k] = flat[idx++];
			}
		}
	}
	return rulesArray;
}

// Optional helper — not currently needed
// Use only if you want to log, store, or inject rule arrays manually

function flattenRules(rulesArray, type) {
	const config = populationConfigs[type];
	const { layers, channels, bitsPerRule } = config;

	const flat = [];

	for (let i = 0; i < layers; i++) {
		for (let j = 0; j < channels; j++) {
			for (let k = 0; k < bitsPerRule; k++) {
				flat.push(rulesArray[i][j][k]);
			}
		}
	}

	return flat;
}

function getrule(){
	let r=[];
	let r2=[];
	for(let i=0;i<18;i++) r[i]=Math.round(Math.random());
	for(let i=0;i<512;i++){
		let q=((i>>4)&1)*8;
		for(let j=0;j<9;j++){
			q+=(i>>j)&1;
		}
		r2[i]=r[q];
	}
	return r2;
}
function padding(array, seed){
	if(seed && array[0].length==1) return seed;
	let temp=[];
	let l2=array[0].length;
	for(let x=0;x<l2;x++){
		temp[x*2+0]=[];
		temp[x*2+1]=[];
		for(let y=0;y<l2;y++){
			temp[x*2+0][y*2+0]=array[0][x][y];
			temp[x*2+0][y*2+1]=array[1][x][y];
			temp[x*2+1][y*2+0]=array[2][x][y];
			temp[x*2+1][y*2+1]=array[3][x][y];
		}
	}
	return temp;
}

function cellular(array, rule) {
	const l2 = array.length;
	const temp = new Array(l2); // Preallocate the result array

	for (let x = 0; x < l2; x++) {
		temp[x] = new Int8Array(l2); // Use Int8Array for better performance
		const xm = (x - 1 + l2) % l2; // Precompute x-1 with periodic boundary
		const xp = (x + 1) % l2; // Precompute x+1 with periodic boundary

		for (let y = 0; y < l2; y++) {
			const ym = (y - 1 + l2) % l2; // Precompute y-1 with periodic boundary
			const yp = (y + 1) % l2; // Precompute y+1 with periodic boundary

			// Combine the 9-cell neighborhood into a single number (q)
			const q =
				(array[xm][ym] << 8) | // Top-left
				(array[x][ym] << 7) | // Top-center
				(array[xp][ym] << 6) | // Top-right
				(array[xm][y] << 5) | // Middle-left
				(array[x][y] << 4) | // Center
				(array[xp][y] << 3) | // Middle-right
				(array[xm][yp] << 2) | // Bottom-left
				(array[x][yp] << 1) | // Bottom-center
				array[xp][yp]; // Bottom-right

			// Apply the rule
			temp[x][y] = rule[q];
		}
	}

	return temp;
}

function draw(array, numberOfEpoch) {
	const size=array.length;
	const canvas = createCanvas(size, size);
	const ctx = canvas.getContext('2d');
	ctx.fillStyle = 'rgb(0,0,0)'
	ctx.fillRect(0, 0, size, size);
	ctx.fillStyle = 'rgb(255,255,255)';
	for (let x = 0; x < size; x++) {
		for (let y = 0; y < size; y++) {
			if(array[x][y])
				ctx.fillRect(x, y, 1, 1);
		}
	}
	ensureStorageDir();
	const outputPath = path.join(storagePath, `${numberOfEpoch}.png`);
	const out = fs.createWriteStream(outputPath);
	const stream = canvas.createPNGStream();
	stream.pipe(out);
	out.on('finish', () => {
		console.log(`Saved image #${numberOfEpoch} to ${outputPath}`);
	});
}

function makeRandomSeed() {
	let seed = [];
	for (let i = 0; i < 4; i++) {
		seed[i] = [];
		for (let j = 0; j < 4; j++) {
			seed[i][j] = Math.round(Math.random());
		}
	}
	return seed;
}

//GENERATOR
function generateFromRule(rulesArray, seed){
	let array;
	let depth=rulesArray.length;
	let temp=[[1]];
	for(let i=0;i<depth;i++){
		array=padding(temp, seed);
		temp=[];
		for(let j=0;j<4;j++){
			temp[j]=cellular(array,rulesArray[i][j]);
		}
	}
	return array;
}





function downsample(array) {
	let size = array.length / 2;
	let result = [];

	for (let x = 0; x < size; x++) {
		result[x] = [];
		for (let y = 0; y < size; y++) {
			let v0 = array[x * 2][y * 2];
			let v1 = array[x * 2 + 1][y * 2];
			let v2 = array[x * 2][y * 2 + 1];
			let v3 = array[x * 2 + 1][y * 2 + 1];

			// symbolic pooling: majority vote
			let sum = v0 + v1 + v2 + v3;
			result[x][y] = sum > 1 ? 1 : 0; 
			//result[x][y] = sum%2;
		}
	}
	return result;
}

//DISCRIMINATOR
function evaluateDiscriminator(input, rulesArray) {
	let current = input;

	for (let i = 0; i < rulesArray.length; i++) {
		let evolved = [];

		for (let j = 0; j < rulesArray[i].length; j++) {
			evolved[j] = cellular(current, rulesArray[i][j]);
		}

		for (let j = 0; j < evolved.length; j++) {
			evolved[j] = downsample(evolved[j]);
		}

		current = padding(evolved);
		
		current = downsample(current);
	}

	return current;
}

function seedBalance(array){
	const size = array.length;
	let sum = 0;
	for (let x = 0; x < size; x++) {
		for (let y = 0; y < size; y++) {
			if(array[x][y]==0) sum++;
		}
	}
	return (sum / (size * size)) * 100;
}

function arrayMatch(array1, array2){
	const size = array1.length;
	let sum = 0;
	for (let x = 0; x < size; x++) {
		for (let y = 0; y < size; y++) {
			if(array1[x][y]==array2[x][y]) sum++;
		}
	}
	return sum;
}

//we don't use it, but keep it
function testFitness(array){
	let fitness=0;
	const size=array.length;

	for (let x = 0; x < size; x++) {
		for (let y = 0; y < size; y++) {
			if(array[x][y]==(Math.floor((x*y) / 8) % 2))
				fitness++;
		}
	}
	return fitness;
}


async function loadImageSample(filePath, size = 512) {
	try {
		const image = await loadImage(filePath);
		const canvas = createCanvas(size, size);
		const ctx = canvas.getContext('2d');

		// Draw and resize the image
		ctx.drawImage(image, 0, 0, size, size);

		const imageData = ctx.getImageData(0, 0, size, size).data;

		const imagesample = [];
		for (let x = 0; x < size; x++) {
			imagesample[x] = [];
			for (let y = 0; y < size; y++) {
				const index = (y * size + x) * 4; // ✅ Row-major indexing
				const r = imageData[index];
				const g = imageData[index + 1];
				const b = imageData[index + 2];

				const gray = 0.299 * r + 0.587 * g + 0.114 * b;
				const noise = Math.random() * 128 - 64;
				imagesample[x][y] = gray > (127 + noise) ? 1 : 0;
			}
		}

		console.log(`✅ Loaded image: ${filePath} (${size}×${size})`);
		return imagesample;
	} catch (error) {
		console.error(`❌ Error loading image ${filePath}:`, error);
		throw error;
	}
}

function outputToResult(array){
	const size = array.length;
	let sum = 0;
	for (let x = 0; x < size; x++) {
		for (let y = 0; y < size; y++) {
			if(array[x][y]) sum++;
		}
	}
	return 100*sum/(size*size);
	
}


async function init() {
	const gType = "Generator";
	const dType = "Discriminator";
	initPopulation(gType);
	initPopulation(dType);
	const g = populations[gType];
	const d = populations[dType];

	const size = 256;
	
	const datasetImage = await loadImageSample(
		path.join(datasetPath, "jess.png"),
		size
	);
	draw(datasetImage, "test");

	// Load (or create) the fitness-history once:
	//let fitnessHistory = loadFitnessHistory();

	for (let epoch = 0; epoch < 500; epoch++) {
		let discriminatorEff = 0;
		let generatorEff = 0;
		
		for (let j = 0; j < PopulationSize; j++) {
			
			const seed = makeRandomSeed();
			const generatorRule = unflattenRules(g.population[j], gType);
			const generatedImage = generateFromRule(generatorRule, seed);

			const discriminatorRule = unflattenRules(d.population[j], dType);

			const scoreReal = outputToResult(
				evaluateDiscriminator(datasetImage, discriminatorRule)
			);
			const scoreFake = outputToResult(
				evaluateDiscriminator(generatedImage, discriminatorRule)
			);

			// discriminator wants real↑ fake↓
			d.fitness[j] = (scoreReal - scoreFake + 100) / 2; // 0–100
			// generator wants fake↑
			g.fitness[j] = scoreFake;

			// efficiency counters
			if (scoreReal > 50 && scoreFake < 50) {
				discriminatorEff++; // D correctly distinguishes both real and fake
			} else if (scoreFake > 50 && scoreReal <= 50) {
				generatorEff++; // G successfully fools D, and D fails at real
			} else {
				// ambiguous or mutual failure — give no one a point
			}
			
			if((epoch%10==0)&&(j==0)){
				draw(generatedImage, epoch);
			}
		}

		// evolve whichever side is lagging
		if (discriminatorEff < PopulationSize * 0.10) evolute(dType);
		if (generatorEff    < PopulationSize * 0.25) evolute(gType);

		console.log(
			`Epoch ${epoch} – GenEff: ${generatorEff}, DiscEff: ${discriminatorEff}`
		);
		
		/*
		// Append a single record per epoch then save
		fitnessHistory.push([generatorEff, discriminatorEff]);
		saveFitnessHistory(fitnessHistory);
		*/
	}
}


//autoencoder
async function init2() {
	const gType = "Generator";
	initPopulation(gType);
	const g = populations[gType];
	
	const dType = "Discriminator";
	initPopulation(dType);
	const d = populations[dType];

	let size=512;
	let imagesample=[]
	
	/*
	for (let x = 0; x < size; x++) {
		imagesample[x]=[]
		for (let y = 0; y < size; y++) {
			imagesample[x][y]=Math.floor((x*y) / 7) % 2;
		}
	}
	*/

	const imagePath = path.join(datasetPath, 'jess.png');
	imagesample = await loadImageSample(imagePath, size);

	draw(imagesample, `testSample`);

	
	let fitnessHistory = loadFitnessHistory(); // Load existing history

	for(let epoch=0;epoch<2000;epoch++){
		imagesample = await loadImageSample(imagePath, size);
		let invalidDiscriminators=0;
		for(let j=0;j<PopulationSize;j++){
			const discriminatorRule = unflattenRules(d.population[j], dType);
			const discriminatorOutput = evaluateDiscriminator(imagesample, discriminatorRule);
			
			//if (j==0) console.log(JSON.stringify(discriminatorOutput));
			
			const generatorRule  = unflattenRules(g.population[j], gType);
			const generatorOutput = generateFromRule(generatorRule , discriminatorOutput);
			
			let fintess=arrayMatch(generatorOutput, imagesample);
			g.fitness[j] = fintess;
			d.fitness[j] = fintess;
			
			const balance = seedBalance(discriminatorOutput);
			if (balance === 0 || balance === 100) {
				invalidDiscriminators++;
			}
			
		}
		
		fitnessHistory.push(g.fitness[0]); // Store best fitness
		saveFitnessHistory(fitnessHistory); // Save after each epoch
		
		if(invalidDiscriminators > PopulationSize / 2){
			
			console.log(`Epoch: ${epoch}. Invalid discriminators: ${invalidDiscriminators}`);
			
			evolute("Discriminator");
			
		}else{
			
			console.log(`Epoch: ${epoch}. Invalid discriminators: ${invalidDiscriminators}. i0 fintess: ${g.fitness[0]}`);
			
			evolute("Generator");
			evolute("Discriminator");
			
			const discriminatorRule = unflattenRules(d.population[0], dType);
			const discriminatorOutput = evaluateDiscriminator(imagesample, discriminatorRule);
			const generatorRule  = unflattenRules(g.population[0], gType);
			const generatorOutput = generateFromRule(generatorRule , discriminatorOutput);
			if(epoch%10==0) draw(generatorOutput, epoch);
		}
	}
	
	
}

// keep this for generator test
function testGenerator() {
	const type = "Generator";
	initPopulation(type);

	const p = populations[type];
	
	for(let i=0;i<5;i++){
		
		for(let j=0;j<PopulationSize;j++){
			const rules = unflattenRules(p.population[j], type);
			const seed = makeRandomSeed();
			const array = generateFromRule(rules, seed);
			p.fitness[j]=testFitness(array);
		}
		
		if(true || i%10==0) console.log(`Epoch ${i}: Fitness = ${p.fitness[0]}`);
		evolute(type);
		
		if(true || i%500==0){
			const bestRules = unflattenRules(p.population[0], type);
			const seed = makeRandomSeed();
			const bestArray = generateFromRule(bestRules, seed);
			draw(bestArray, i);
		}
		
	}
	
}


// Initialize simulator
init();

module.exports = { init };


