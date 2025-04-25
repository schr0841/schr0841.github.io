// Get canvas element and context
const canvas = document.getElementById('digit-canvas');
const ctx = canvas.getContext('2d');
const predictButton = document.getElementById('predict-button');
const clearButton = document.getElementById('clear-button');
const predictionOutput = document.getElementById('prediction-output');
const statusMessage = document.getElementById('status-message');

let isDrawing = false;
let model; // Variable to hold the created TensorFlow.js model

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
    if (model) { // Only update status if model has been created
       statusMessage.textContent = 'Model structure defined (untrained). Ready to predict.';
    } else {
       statusMessage.textContent = 'Creating model structure...';
    }
}

clearButton.addEventListener('click', clearCanvas);
predictButton.addEventListener('click', predictDigit);

// --- TensorFlow.js Model Creation ---

// Function to define the CNN model structure
function createModel() {
    statusMessage.textContent = 'Creating model structure...';
    try {
        // Define a sequential model
        model = tf.sequential();

        // Input shape: 28x28 pixels, 1 color channel (grayscale)
        const inputShape = [28, 28, 1];

        // Convolutional Layer 1
        // 32 filters, 3x3 kernel size, ReLU activation
        model.add(tf.layers.conv2d({
            inputShape: inputShape,
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
        }));

        // Max Pooling Layer 1
        // 2x2 pool size
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Convolutional Layer 2
        // 64 filters, 3x3 kernel size, ReLU activation
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }));

        // Max Pooling Layer 2
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Flatten Layer
        // Flattens the output of the pooling layer to feed into dense layers
        model.add(tf.layers.flatten());

        // Dense Layer (Fully Connected)
        // 128 units, ReLU activation
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));

        // Output Layer
        // 10 units (for digits 0-9), Softmax activation for probability distribution
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

        // Compile the model - necessary even if not training here
        // Optimizer and loss are needed for compilation but won't be used
        // unless you train the model later.
        model.compile({
            optimizer: tf.train.adam(), // Example optimizer
            loss: 'categoricalCrossentropy', // Example loss function
            metrics: ['accuracy'], // Example metric
        });

        model.summary(); // Log model structure to the console (optional)

        statusMessage.textContent = 'Model structure defined (untrained). Ready to predict!';
        predictButton.disabled = false; // Enable predict button
        console.log('Model structure created successfully.');

    } catch (error) {
        statusMessage.textContent = 'Error creating model structure. Check console.';
        console.error('Error creating model:', error);
        predictButton.disabled = true;
    }
}

// Function to preprocess the canvas image and predict
async function predictDigit() {
    if (!model) {
        statusMessage.textContent = 'Model structure not created yet.';
        return;
    }

    statusMessage.textContent = 'Processing and predicting (using untrained model)...';
    predictionOutput.textContent = '...';

    // Preprocess the canvas data using tf.tidy for memory management
    const imageData = preprocessCanvas(canvas);

    // Make prediction
    try {
        // Predict using the UNTRAINED model. Output will be based on initial random weights.
        const predictionTensor = model.predict(imageData);

        const probabilities = await predictionTensor.data();
        const predictedClass = predictionTensor.argMax(1).dataSync()[0];

        // The prediction will likely be random!
        predictionOutput.textContent = `Predicted Digit (Untrained): ${predictedClass}`;
        statusMessage.textContent = 'Prediction complete (untrained model).';

        // Dispose tensors
        imageData.dispose(); // Disposed by tidy
        predictionTensor.dispose();

    } catch (error) {
        statusMessage.textContent = 'Error during prediction. Check console.';
        predictionOutput.textContent = 'Prediction Failed';
        console.error('Prediction error:', error);
        // Ensure imageData is disposed even if prediction fails
        imageData.dispose(); // Handled by tidy, but safe to leave
    }
}

// Function to preprocess the canvas image (No changes needed here)
function preprocessCanvas(canvasInput) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(canvasInput, 1);
        const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat();
        const normalized = resized.div(tf.scalar(255.0));
        const inverted = tf.scalar(1.0).sub(normalized);
        const batched = inverted.expandDims(0);
        return batched;
    });
}

// --- Initial Setup ---
predictButton.disabled = true; // Disable button until model is created
clearCanvas(); // Ensure canvas is clear initially
createModel(); // Create the model structure when the script runs

