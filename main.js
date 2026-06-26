import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video = document.getElementById("webcam")
const canvas = document.getElementById("output")
const ctx = canvas.getContext("2d")

canvas.width = window.innerWidth
canvas.height = window.innerHeight

const scene = new THREE.Scene()

const camera3D = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
)
camera3D.position.z = 2

const CAMERA_SENSIVITY = 5

const renderer = new THREE.WebGLRenderer({alpha: true})
renderer.setClearColor(0x000000, 0)
renderer.setSize(window.innerWidth, window.innerHeight)

document.getElementById("three-container").appendChild(renderer.domElement)

const geometry = new THREE.BoxGeometry()
const material = new THREE.MeshNormalMaterial()
const cube = new THREE.Mesh(geometry, material)
scene.add(cube)

let prev = null
let velocityX = 0
let velocityY = 0
let velCamZ = 0
const DAMPING = 0.92
const MOMENTUM_SCALE = 0.4 

function getDelta(current) {
    if (!prev) {
        prev = current
        return { dx: 0, dy: 0 }
    }
    let dx = current.x - prev.x
    let dy = current.y - prev.y
    prev = current
    return { dx, dy }
}

function animate() {
    requestAnimationFrame(animate)

    cube.rotation.x += velocityY
    cube.rotation.y += velocityX
    
    camera3D.position.z += velCamZ

    velocityX *= DAMPING
    velocityY *= DAMPING
    velCamZ *= DAMPING

    if (Math.abs(velocityX) < 0.0001) velocityX = 0
    if (Math.abs(velocityY) < 0.0001) velocityY = 0
    if (Math.abs(velCamZ) < 0.0001) velCamZ = 0

    renderer.render(scene, camera3D)
}
animate()

let handLandmarker = undefined
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm")
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.7,
        minHandPresenceConfidence: 0.7
    })
    console.log(handLandmarker);
}
createHandLandmarker()

let tfModel = undefined
const loadModel = async () => {
    tfModel = await tf.loadLayersModel("tfjs/model.json")
    console.log(tfModel);
}
loadModel()

const LABELS = ["Tunjuk", "Kepal", "Hi"]

function normalize(coords) {
    let reshaped = []

    // Reshape
    for (let i = 0; i < 21; i++) {
        reshaped.push([
            coords[i * 3],
            coords[i * 3 + 1],
            coords[i * 3 + 2]
        ])
    }

    let wrist = reshaped[0]
    reshaped = reshaped.map(p => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]])

    let maxVal = Math.max(...reshaped.flat().map(Math.abs))
    reshaped = reshaped.map(p => [p[0] / maxVal, p[1] / maxVal, p[2] / maxVal])

    return reshaped.flat()
}

// const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia
// if (hasGetUserMedia()) {} // Something

navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {
    video.srcObject = stream
    video.addEventListener("loadeddata", predictWebcam)
})

let lastVideoTime = -1
let results = undefined
console.log(video)

async function predictWebcam() {
    let startTimeMs = performance.now()
    if (lastVideoTime != video.currentTime) {
        lastVideoTime = video.currentTime
        results = handLandmarker.detectForVideo(video, startTimeMs)
    }

    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: "#FFFFFF" , lineWidth: 3})
            drawLandmarks(ctx, landmarks, { color: "#FF0000", radius: 2})
            // console.log(landmarks)

            let coords = []
            for (let lm of landmarks) {
                coords.push(lm.x, lm.y, lm.z)
            }
            let input = normalize(coords)
            const tensor = tf.tensor([input])

            const pred = tfModel.predict(tensor)
            const data = await pred.data()

            let class_id = data.indexOf(Math.max(...data))
            let confidence = Math.max(...data)
            let label = LABELS[class_id]

            console.log(`${label} (${confidence})`);
            if (label === "Tunjuk") {
                const tip = {x: coords[8 * 3], y: coords[8 * 3 + 1]}
                let {dx, dy} = getDelta(tip)

                if (Math.abs(dx) < 0.002) dx = 0
                if (Math.abs(dy) < 0.002) dy = 0

                velocityX = velocityX * 0.6 + dx * CAMERA_SENSIVITY * MOMENTUM_SCALE
                velocityY = velocityY * 0.6 + dy * CAMERA_SENSIVITY * MOMENTUM_SCALE
            } else {
                prev = null
            }

            if (label === "Hi") {
                // camera3D.position.z += 0.1
                velCamZ = velCamZ * 0.6 + 0.025 * MOMENTUM_SCALE
            }

            if (label === "Kepal") {
                // camera3D.position.z -= 0.1
                velCamZ = velCamZ * 0.6 - 0.025 * MOMENTUM_SCALE
            }
            
            tensor.dispose()
            pred.dispose()
        }
    }
    ctx.restore()

    window.requestAnimationFrame(predictWebcam)
}

// setInterval(() => {
//     cube.rotation.x += 0.1
//     cube.rotation.y += 0.1
// }, 50)