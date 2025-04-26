// Get canvas element and context
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predict-button');
const clearButton = document.getElementById('clear-button');
const predictionOutput = document.getElementById('prediction-output');
const statusMessage = document.getElementById('status-message');

let isDrawing = false;
let model; // Variable to hold the loaded TensorFlow.js model (GraphModel)

// Set drawing style
ctx.lineWidth = 15; // Adjust thickness as needed
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

// --- Drawing Event Listeners ---

function startDrawing(e) {
    isDrawing = true;
    draw(e); // Start drawing immediately at the touch/click point
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath(); // Reset path when mouse is lifted
}

function draw(e) {
    if (!isDrawing) return;

    // Prevent scrolling while drawing on touch devices
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    let x, y;

    // Handle both mouse and touch events
    if (e.touches && e.touches.length > 0) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath(); // Begin new path segment
    ctx.moveTo(x, y); // Move to the current point
}

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing); // Stop drawing if mouse leaves canvas
canvas.addEventListener('mousemove', draw);

// Touch events
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchcancel', stopDrawing);
canvas.addEventListener('touchmove', draw);


// --- Button Actions ---

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictionOutput.textContent = 'Draw a digit and click Predict.';
    if (model) { // Only update status if model has been loaded/attempted
       statusMessage.textContent = 'Model ready.';
    } else {
       statusMessage.textContent = 'Loading model...';
    }
}

clearButton.addEventListener('click', clearCanvas);
predictButton.addEventListener('click', predictDigit);

// --- TensorFlow.js Model Loading and Prediction ---

// Function to load the model using loadGraphModel
async function loadModel() {
    statusMessage.textContent = 'Loading model (GraphModel)...'; // Updated message
    try {
        // Load model from the 'model' folder relative to the HTML page
        model = await tf.loadGraphModel('model/model.json'); // Use loadGraphModel

        // Check if model loaded (basic check for GraphModel)
        if (!model || !model.executor) { // GraphModel has an 'executor' property
            throw new Error("Model structure seems invalid after loading.");
        }

        statusMessage.textContent = 'Model loaded successfully. Ready to predict!';
        predictButton.disabled = false; // Enable predict button
        console.log('GraphModel loaded successfully.');

    } catch (error) {
        statusMessage.textContent = `Error loading model from 'model/model.json'. Check path and console.`;
        console.error('Error loading model:', error); // Log the actual error object
        predictButton.disabled = true; // Disable predict button if model fails
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

    // Preprocess the canvas data using tf.tidy for memory management
    // imageData is created inside tf.tidy in preprocessCanvas and should be disposed there
    const imageData = preprocessCanvas(canvas);
    let predictionTensor; // Declare predictionTensor outside try block

    try {
        // Make prediction using predict or executeAsync
        if (typeof model.predict === 'function') {
             predictionTensor = model.predict(imageData);
        } else if (typeof model.executeAsync === 'function') {
             predictionTensor = await model.executeAsync(imageData);
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
        // ** Ensure tensors are disposed even if errors occur **
        // imageData should be disposed by tf.tidy in preprocessCanvas, but explicit disposal is safe
        if (imageData) {
             imageData.dispose();
             // console.log('imageData disposed'); // Uncomment for debugging
        }
        // Dispose predictionTensor if it exists
        if (predictionTensor) {
             predictionTensor.dispose();
             // console.log('predictionTensor disposed'); // Uncomment for debugging
        }
    }
}

// Function to preprocess the canvas image
function preprocessCanvas(canvasInput) {
    // Use tf.tidy to automatically dispose intermediate tensors created ONLY within this function
    return tf.tidy(() => {
        // 1. Get image data from canvas and convert to tensor (grayscale)
        let tensor = tf.browser.fromPixels(canvasInput, 1);

        // 2. Resize the image to 28x28 pixels using bilinear interpolation
        const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat();

        // 3. Normalize the image (pixels 0-255 to 0-1). Invert colors.
        const normalized = resized.div(tf.scalar(255.0));
        const inverted = tf.scalar(1.0).sub(normalized);

        // 4. Reshape to the expected batch format [batch_size, height, width, channels]
        const batched = inverted.expandDims(0); // [1, 28, 28, 1]

        // console.log('Preprocessed tensor created'); // Uncomment for debugging
        return batched; // tf.tidy will dispose intermediate tensors (tensor, resized, normalized, inverted)
                        // but NOT the returned 'batched' tensor.
    });
}

// --- Initial Setup ---
predictButton.disabled = true; // Disable button until model is loaded
clearCanvas(); // Ensure canvas is clear initially
loadModel();   // Start loading the model when the script runs
