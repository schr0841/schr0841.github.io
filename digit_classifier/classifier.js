// Get canvas element and context
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predict-button');
const clearButton = document.getElementById('clear-button');
const predictionOutput = document.getElementById('prediction-output');
const statusMessage = document.getElementById('status-message');
// Removed debug canvas variable

let isDrawing = false;
let model; // Variable to hold the loaded TensorFlow.js model (GraphModel)

// --- Drawing Style Setup ---
function setupDrawingStyle() {
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.fillStyle = 'white'; // Add fill style for clearing
}

// --- Drawing Event Listeners ---
// (No changes needed in drawing listeners)
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}
function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}
function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    setupDrawingStyle(); // Re-apply style
    const rect = canvas.getBoundingClientRect();
    let x, y;
    if (e.touches && e.touches.length > 0) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchcancel', stopDrawing);
canvas.addEventListener('touchmove', draw);

// --- Button Actions ---
function clearCanvas() {
    setupDrawingStyle();
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    // Removed clearing of debug canvas
    predictionOutput.textContent = 'Draw a digit and click Predict.';
    if (model) {
       statusMessage.textContent = 'Model ready.';
    } else {
       statusMessage.textContent = 'Loading model...';
    }
}
clearButton.addEventListener('click', clearCanvas);
predictButton.addEventListener('click', predictDigit);

// --- TensorFlow.js Model Loading and Prediction ---
// (No changes needed in loadModel)
async function loadModel() {
    statusMessage.textContent = 'Loading model (GraphModel)...';
    try {
        model = await tf.loadGraphModel('../digit_classifier/model/model.json');
        if (!model || !model.executor) {
            throw new Error("Model structure seems invalid after loading.");
        }
        statusMessage.textContent = 'Model loaded successfully. Ready to predict!';
        predictButton.disabled = false;
        console.log('GraphModel loaded successfully.');
    } catch (error) {
        statusMessage.textContent = `Error loading model from 'digit_classifier/model/model.json'. Check path and console.`;
        console.error('Error loading model:', error);
        predictButton.disabled = true;
    }
}

// Function to preprocess the canvas image and predict
async function predictDigit() {
    if (!model) {
        statusMessage.textContent = 'Model not loaded yet.';
        return;
    }

    statusMessage.textContent = 'Processing and predicting...';
    predictionOutput.textContent = '...';

    // imageData will be normalized (black=0, white=1)
    const imageData = preprocessCanvas(canvas);
    let predictionTensor;
    // Removed imageTensorForViz variable

    // --- Debug Visualization Removed ---

    try {
        // Make prediction using predict or executeAsync
        if (typeof model.predict === 'function') {
             predictionTensor = model.predict(imageData); // Input is now black=0, white=1
        } else if (typeof model.executeAsync === 'function') {
             predictionTensor = await model.executeAsync(imageData); // Input is now black=0, white=1
        } else {
            throw new Error("Model does not have predict or executeAsync method.");
        }

        // Get the highest probability index (predicted class) asynchronously
        const predictionResult = predictionTensor.argMax(1);
        const predictedClassData = await predictionResult.data(); // Use async data()
        const predictedClass = predictedClassData[0];

        // Dispose the intermediate tensor from argMax
        predictionResult.dispose();

        predictionOutput.textContent = `Predicted Digit: ${predictedClass}`;
        statusMessage.textContent = 'Prediction complete.';

    } catch (error) {
        statusMessage.textContent = 'Error during prediction. Check console.';
        predictionOutput.textContent = 'Prediction Failed';
        console.error('Prediction error:', error);
    } finally {
        // Ensure tensors are disposed even if errors occur
        if (imageData) {
             imageData.dispose();
        }
        if (predictionTensor) {
             predictionTensor.dispose();
        }
        // Removed disposal of imageTensorForViz
    }
}

// Function to preprocess the canvas image
function preprocessCanvas(canvasInput) {
    return tf.tidy(() => {
        // 1. Get pixel data using the canvas 2D context
        const ctx = canvasInput.getContext('2d');
        const imgData = ctx.getImageData(0, 0, canvasInput.width, canvasInput.height);

        // 2. Create tensor from the ImageData.data (Uint8ClampedArray)
        let rawTensorRGBA = tf.tensor(imgData.data, [canvasInput.height, canvasInput.width, 4], 'int32');

        // 3. Slice off the alpha channel and convert to float
        let rawTensorRGB = rawTensorRGBA.slice([0, 0, 0], [canvasInput.height, canvasInput.width, 3]).toFloat();

        // 4. Convert RGB to Grayscale using luminance weights
        const rgbWeights = tf.tensor1d([0.299, 0.587, 0.114]);
        let grayscaleTensor = rawTensorRGB.mul(rgbWeights).sum(2).expandDims(2); // Shape [height, width, 1]

        // 5. Resize the image to 28x28 pixels
        const resized = tf.image.resizeBilinear(grayscaleTensor, [28, 28]); // Input is now [0-255] grayscale

        // 6. Normalize the image (pixels 0-255 to 0-1).
        const normalized = resized.div(tf.scalar(255.0)); // Now [0-1], black=0, white=1

        // ** Keep inversion step - seems necessary for model **
        const inverted = tf.scalar(1.0).sub(normalized); // Now [0-1], black=1, white=0

        // 7. Reshape to the expected batch format [batch_size, height, width, channels]
        const batched = inverted.expandDims(0); // Use inverted [1, 28, 28, 1]

        // Dispose intermediate tensors that are not returned by tf.tidy
        rawTensorRGBA.dispose();
        rawTensorRGB.dispose();

        return batched; // Return inverted (black=1, white=0)
    });
}

// --- Initial Setup ---
setupDrawingStyle(); // Call initially
// ** Fill background white initially **
ctx.fillRect(0, 0, canvas.width, canvas.height);
predictButton.disabled = true; // Disable button until model is loaded
loadModel();   // Start loading the model when the script runs
