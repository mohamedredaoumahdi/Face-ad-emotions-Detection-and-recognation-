import AVFoundation
import UIKit
import Vision
import CoreML

class LiveFeedViewController: UIViewController {
    
    // MARK: - Properties
    
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var faceLayers: [CAShapeLayer] = []
    private var frameCounter = 0
    private var currentDevice: AVCaptureDevice?
    private var showLandmarks = true
    
    // Track detected emotions for stability
    private var emotionHistory: [String] = []
    private let emotionHistoryMaxSize = 5
    
    // MARK: - UI Components
    
    private let emotionLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 20, weight: .bold)
        label.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        label.layer.cornerRadius = 12
        label.layer.masksToBounds = true
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let emojiLabel: UILabel = {
        let label = UILabel()
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 40)
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let confidenceView: UIProgressView = {
        let progress = UIProgressView(progressViewStyle: .bar)
        progress.trackTintColor = UIColor.lightGray.withAlphaComponent(0.5)
        progress.progressTintColor = UIColor(named: "AccentColor") ?? .systemPurple
        progress.layer.cornerRadius = 4
        progress.clipsToBounds = true
        progress.translatesAutoresizingMaskIntoConstraints = false
        return progress
    }()
    
    private let flipCameraButton: UIButton = {
        let button = UIButton(type: .system)
        button.setImage(UIImage(systemName: "camera.rotate"), for: .normal)
        button.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        button.tintColor = .white
        button.layer.cornerRadius = 25
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    private let settingsButton: UIButton = {
        let button = UIButton(type: .system)
        button.setImage(UIImage(systemName: "gearshape"), for: .normal)
        button.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        button.tintColor = .white
        button.layer.cornerRadius = 25
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    private let infoPanel: UIView = {
        let view = UIView()
        view.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        view.layer.cornerRadius = 16
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }()
    
    private lazy var backButton: UIButton = {
        let button = UIButton(type: .system)
        button.setImage(UIImage(systemName: "chevron.left"), for: .normal)
        button.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        button.tintColor = .white
        button.layer.cornerRadius = 20
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addTarget(self, action: #selector(backButtonTapped), for: .touchUpInside)
        return button
    }()
    
    // MARK: - Lifecycle Methods
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupCamera()
        
        // Start session in background to avoid UI freezing
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        captureSession.stopRunning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Start capture session if it was stopped
        if !captureSession.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession.startRunning()
            }
        }
    }
    
    // MARK: - Setup Methods
    
    private func setupUI() {
        navigationController?.setNavigationBarHidden(true, animated: false)
        
        // Add preview layer
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        // Add UI components
        view.addSubview(emotionLabel)
        view.addSubview(infoPanel)
        infoPanel.addSubview(emojiLabel)
        infoPanel.addSubview(confidenceView)
        view.addSubview(flipCameraButton)
        view.addSubview(settingsButton)
        view.addSubview(backButton)
        
        // Set up constraints
        NSLayoutConstraint.activate([
            // Emotion label at top
            emotionLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            emotionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emotionLabel.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -40),
            emotionLabel.heightAnchor.constraint(equalToConstant: 40),
            emotionLabel.leadingAnchor.constraint(greaterThanOrEqualTo: view.leadingAnchor, constant: 20),
            emotionLabel.trailingAnchor.constraint(lessThanOrEqualTo: view.trailingAnchor, constant: -20),
            
            // Info panel at bottom
            infoPanel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            infoPanel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            infoPanel.widthAnchor.constraint(equalToConstant: 200),
            infoPanel.heightAnchor.constraint(equalToConstant: 120),
            
            // Emoji label in info panel
            emojiLabel.topAnchor.constraint(equalTo: infoPanel.topAnchor, constant: 20),
            emojiLabel.centerXAnchor.constraint(equalTo: infoPanel.centerXAnchor),
            emojiLabel.widthAnchor.constraint(equalToConstant: 60),
            emojiLabel.heightAnchor.constraint(equalToConstant: 60),
            
            // Confidence view in info panel
            confidenceView.topAnchor.constraint(equalTo: emojiLabel.bottomAnchor, constant: 10),
            confidenceView.leadingAnchor.constraint(equalTo: infoPanel.leadingAnchor, constant: 20),
            confidenceView.trailingAnchor.constraint(equalTo: infoPanel.trailingAnchor, constant: -20),
            confidenceView.heightAnchor.constraint(equalToConstant: 8),
            
            // Flip camera button at right edge
            flipCameraButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            flipCameraButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            flipCameraButton.widthAnchor.constraint(equalToConstant: 50),
            flipCameraButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Settings button
            settingsButton.topAnchor.constraint(equalTo: flipCameraButton.bottomAnchor, constant: 20),
            settingsButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            settingsButton.widthAnchor.constraint(equalToConstant: 50),
            settingsButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Back button
            backButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            backButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            backButton.widthAnchor.constraint(equalToConstant: 40),
            backButton.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        // Set up button actions
        flipCameraButton.addTarget(self, action: #selector(flipCamera), for: .touchUpInside)
        settingsButton.addTarget(self, action: #selector(showSettings), for: .touchUpInside)
        
        // Initial UI state
        emotionLabel.text = "Looking for faces..."
        emojiLabel.text = "🔍"
        confidenceView.progress = 0.0
    }
    
    private func setupCamera() {
        captureSession.sessionPreset = .high
        
        // Start with front camera
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .front
        )
        
        guard let device = deviceDiscoverySession.devices.first else {
            print("No camera device found")
            return
        }
        
        currentDevice = device
        
        do {
            // Configure camera for better face detection
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            device.unlockForConfiguration()
            
            let deviceInput = try AVCaptureDeviceInput(device: device)
            if captureSession.canAddInput(deviceInput) {
                captureSession.addInput(deviceInput)
                setupVideoOutput()
            }
        } catch {
            print("Error setting up camera: \(error.localizedDescription)")
        }
    }
    
    private func setupVideoOutput() {
        videoDataOutput.videoSettings = [
            (kCVPixelBufferPixelFormatTypeKey as String): Int(kCVPixelFormatType_32BGRA)
        ]
        
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.queue"))
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        if let connection = videoDataOutput.connection(with: .video) {
            connection.videoOrientation = .portrait
            
            // For front camera, enable mirroring
            if currentDevice?.position == .front {
                connection.isVideoMirrored = true
            }
                    
            // Fix for orientation
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
    }
    
    // MARK: - Face Detection and Emotion Recognition
    
    private func detectFaces(in imageBuffer: CVPixelBuffer) {
        let faceDetectionRequest = VNDetectFaceLandmarksRequest { [weak self] request, error in
            guard let self = self else { return }
            
            if let error = error {
                print("Face detection error: \(error.localizedDescription)")
                return
            }
            
            DispatchQueue.main.async {
                // Clear previous drawings
                self.clearFaceLayersAndUpdateUI()
                
                if let observations = request.results as? [VNFaceObservation], !observations.isEmpty {
                    // Draw faces
                    self.handleFaceDetectionObservations(observations: observations)
                    
                    // Process the first face for emotion (typically the main one)
                    if let firstFace = observations.first {
                        self.processFaceForEmotion(firstFace, from: imageBuffer)
                    }
                } else {
                    // No faces detected
                    self.updateUI(emotion: nil, confidence: 0)
                }
            }
        }
        
        // Use leftMirrored orientation for front camera - this matches your original code
        let orientation: CGImagePropertyOrientation = .leftMirrored
        
        let handler = VNImageRequestHandler(
            cvPixelBuffer: imageBuffer,
            orientation: orientation,
            options: [:]
        )
        
        do {
            try handler.perform([faceDetectionRequest])
        } catch {
            print("Failed to perform face detection: \(error)")
        }
    }
    
    private func processFaceForEmotion(_ face: VNFaceObservation, from imageBuffer: CVPixelBuffer) {
        // Extract and crop the face region for better emotion detection
        let faceImage = cropFace(from: imageBuffer, observation: face)
        
        if let emotion = detectEmotion(for: faceImage) {
            updateEmotionHistory(emotion: emotion.class, confidence: emotion.confidence)
        }
    }
    
    private func cropFace(from pixelBuffer: CVPixelBuffer, observation: VNFaceObservation) -> CVPixelBuffer {
        // For simplicity, we'll just use the original buffer
        // In a production app, you would crop the face region here
        return pixelBuffer
    }
    
    private func detectEmotion(for imageBuffer: CVPixelBuffer) -> (class: String, confidence: Float)? {
        do {
            guard let modelURL = Bundle.main.url(forResource: "EmotionClassificationModel", withExtension: "mlmodelc") else {
                print("Error: Model file not found in bundle")
                return nil
            }
            
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let emotionClassifier = try EmotionClassificationModel(model: model)
            
            let prediction = try emotionClassifier.prediction(image: imageBuffer)
            
            // Get the confidence level for the predicted class
            let confidence = prediction.classLabelProbs[prediction.classLabel] ?? 0
            
            return (prediction.classLabel, Float(confidence))
        } catch {
            print("Emotion detection error: \(error)")
            return nil
        }
    }
    
    private func updateEmotionHistory(emotion: String, confidence: Float) {
        // Add to history
        emotionHistory.append(emotion)
        
        // Keep history at max size
        if emotionHistory.count > emotionHistoryMaxSize {
            emotionHistory.removeFirst()
        }
        
        // Find most common emotion in history (simple mode algorithm)
        var emotionCounts: [String: Int] = [:]
        for historyEmotion in emotionHistory {
            emotionCounts[historyEmotion, default: 0] += 1
        }
        
        if let (stableEmotion, _) = emotionCounts.max(by: { $0.value < $1.value }) {
            // Only update UI if we have a stable emotion prediction
            updateUI(emotion: stableEmotion, confidence: confidence)
        }
    }
    
    private func updateUI(emotion: String?, confidence: Float) {
        // If no emotion detected
        guard let emotion = emotion else {
            emotionLabel.text = "No face detected"
            emojiLabel.text = "🔍"
            confidenceView.progress = 0.0
            return
        }
        
        // Update emotion label with confidence
        let confidencePercentage = Int(confidence * 100)
        emotionLabel.text = "\(emotion) (\(confidencePercentage)%)"
        
        // Update emoji based on emotion
        let emoji = emojiForEmotion(emotion)
        emojiLabel.text = emoji
        
        // Update color based on emotion
        emotionLabel.backgroundColor = colorForEmotion(emotion).withAlphaComponent(0.7)
        confidenceView.progressTintColor = colorForEmotion(emotion)
        
        // Update confidence bar
        confidenceView.progress = confidence
        
        // Provide haptic feedback on significant emotion changes
        if emotionHistory.count > 1 && emotionHistory[emotionHistory.count - 1] != emotionHistory[emotionHistory.count - 2] {
            let generator = UIImpactFeedbackGenerator(style: .medium)
            generator.impactOccurred()
        }
    }
    
    private func emojiForEmotion(_ emotion: String) -> String {
        switch emotion.lowercased() {
        case "happy", "happiness":
            return "😄"
        case "sad", "sadness":
            return "😢"
        case "angry", "anger":
            return "😠"
        case "surprised", "surprise":
            return "😮"
        case "fear", "fearful":
            return "😨"
        case "disgust", "disgusted":
            return "🤢"
        case "neutral":
            return "😐"
        case "contempt":
            return "😒"
        default:
            return "🤔"
        }
    }
    
    private func colorForEmotion(_ emotion: String) -> UIColor {
        switch emotion.lowercased() {
        case "happy", "happiness":
            return UIColor(red: 1.0, green: 0.92, blue: 0.23, alpha: 1.0) // Yellow
        case "sad", "sadness":
            return UIColor(red: 0.13, green: 0.59, blue: 0.95, alpha: 1.0) // Blue
        case "angry", "anger":
            return UIColor(red: 0.96, green: 0.26, blue: 0.21, alpha: 1.0) // Red
        case "surprised", "surprise":
            return UIColor(red: 1.0, green: 0.58, blue: 0.0, alpha: 1.0)  // Orange
        case "fear", "fearful":
            return UIColor(red: 0.5, green: 0.0, blue: 0.5, alpha: 1.0)   // Purple
        case "disgust", "disgusted":
            return UIColor(red: 0.0, green: 0.78, blue: 0.33, alpha: 1.0) // Green
        case "neutral":
            return UIColor(red: 0.74, green: 0.74, blue: 0.74, alpha: 1.0) // Gray
        case "contempt":
            return UIColor(red: 0.39, green: 0.39, blue: 0.39, alpha: 1.0) // Dark Gray
        default:
            return UIColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)    // Default Gray
        }
    }
    
    private func clearFaceLayersAndUpdateUI() {
        // Remove existing face layers
        faceLayers.forEach { $0.removeFromSuperlayer() }
        faceLayers.removeAll()
    }
    
    private func handleFaceDetectionObservations(observations: [VNFaceObservation]) {
        for observation in observations {
            // Convert normalized rect to match preview layer coordinates
            let faceRect = previewLayer.layerRectConverted(fromMetadataOutputRect: observation.boundingBox)
            
            // Create face bounding box
            let faceLayer = createFaceLayer(for: faceRect)
            faceLayers.append(faceLayer)
            view.layer.addSublayer(faceLayer)
            
            // Add landmarks if enabled
            if showLandmarks, let landmarks = observation.landmarks {
                // Use the exact method from your original code
                if let leftEye = landmarks.leftEye {
                    self.handleLandmark(leftEye, faceBoundingBox: faceRect)
                }
                if let leftEyebrow = landmarks.leftEyebrow {
                    self.handleLandmark(leftEyebrow, faceBoundingBox: faceRect)
                }
                if let rightEye = landmarks.rightEye {
                    self.handleLandmark(rightEye, faceBoundingBox: faceRect)
                }
                if let rightEyebrow = landmarks.rightEyebrow {
                    self.handleLandmark(rightEyebrow, faceBoundingBox: faceRect)
                }
                
                if let nose = landmarks.nose {
                    self.handleLandmark(nose, faceBoundingBox: faceRect)
                }
                
                if let outerLips = landmarks.outerLips {
                    self.handleLandmark(outerLips, faceBoundingBox: faceRect)
                }
                if let innerLips = landmarks.innerLips {
                    self.handleLandmark(innerLips, faceBoundingBox: faceRect)
                }
            }
        }
    }
    
    private func createFaceLayer(for rect: CGRect) -> CAShapeLayer {
        let faceLayer = CAShapeLayer()
        faceLayer.path = UIBezierPath(roundedRect: rect, cornerRadius: 10).cgPath
        faceLayer.fillColor = UIColor.clear.cgColor
        faceLayer.strokeColor = UIColor.yellow.cgColor
        faceLayer.lineWidth = 3
        
        // Add animation
        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = 0.5
        animation.toValue = 1.0
        animation.duration = 0.3
        faceLayer.add(animation, forKey: "fadeIn")
        
        return faceLayer
    }
    
    // Using the exact same method from your original code
    private func handleLandmark(_ eye: VNFaceLandmarkRegion2D, faceBoundingBox: CGRect) {
        let landmarkPath = CGMutablePath()
        let landmarkPathPoints = eye.normalizedPoints
            .map({ eyePoint in
                CGPoint(
                    x: eyePoint.y * faceBoundingBox.height + faceBoundingBox.origin.x,
                    y: eyePoint.x * faceBoundingBox.width + faceBoundingBox.origin.y)
            })
        
        landmarkPath.addLines(between: landmarkPathPoints)
        landmarkPath.closeSubpath()
        let landmarkLayer = CAShapeLayer()
        landmarkLayer.path = landmarkPath
        landmarkLayer.fillColor = UIColor.clear.cgColor
        landmarkLayer.strokeColor = UIColor.green.cgColor
        
        self.faceLayers.append(landmarkLayer)
        self.view.layer.addSublayer(landmarkLayer)
    }
    
    // MARK: - Actions
    
    @objc private func flipCamera() {
        // Verify current device position
        guard let currentDevice = self.currentDevice else { return }
        
        // Determine new position
        let newPosition: AVCaptureDevice.Position = currentDevice.position == .front ? .back : .front
        
        // Find device with new position
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: newPosition
        )
        
        guard let newDevice = deviceDiscoverySession.devices.first else { return }
        
        // Start configuration
        captureSession.beginConfiguration()
        
        // Remove current input
        if let currentInput = captureSession.inputs.first {
            captureSession.removeInput(currentInput)
        }
        
        // Add new input
        do {
            let newInput = try AVCaptureDeviceInput(device: newDevice)
            if captureSession.canAddInput(newInput) {
                captureSession.addInput(newInput)
                self.currentDevice = newDevice
            }
        } catch {
            print("Error switching cameras: \(error.localizedDescription)")
        }
        
        // Update video connection (mirroring only for front camera)
        if let connection = videoDataOutput.connection(with: .video) {
            connection.isVideoMirrored = (newPosition == .front)
        }
        
        // Commit configuration
        captureSession.commitConfiguration()
        
        // Add a smooth transition animation
        let transition = CATransition()
        transition.duration = 0.5
        transition.type = .fade
        self.view.layer.add(transition, forKey: nil)
        
        // Provide haptic feedback
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
    }
    
    @objc private func showSettings() {
        let alertController = UIAlertController(
            title: "Settings",
            message: nil,
            preferredStyle: .actionSheet
        )
        
        // Toggle landmarks
        let landmarksTitle = showLandmarks ? "Hide Facial Landmarks" : "Show Facial Landmarks"
        alertController.addAction(UIAlertAction(title: landmarksTitle, style: .default) { [weak self] _ in
            self?.showLandmarks.toggle()
        })
        
        // Toggle processing frequency
        alertController.addAction(UIAlertAction(title: "Toggle Processing Speed", style: .default) { [weak self] _ in
            // Adjust frame processing rate
            if self?.frameCounter == 5 {
                self?.frameCounter = 2 // Process more frames (faster)
            } else {
                self?.frameCounter = 5 // Process fewer frames (slower but more efficient)
            }
        })
        
        // About model
        alertController.addAction(UIAlertAction(title: "About Emotion Model", style: .default) { [weak self] _ in
            let infoAlert = UIAlertController(
                title: "Emotion Classification Model",
                message: "This app uses a custom trained CoreML model for emotion recognition. The model can identify 7 emotions: Happy, Sad, Angry, Surprised, Fear, Disgust, and Neutral.",
                preferredStyle: .alert
            )
            infoAlert.addAction(UIAlertAction(title: "OK", style: .default))
            self?.present(infoAlert, animated: true)
        })
        
        alertController.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        
        present(alertController, animated: true)
    }
    
    @objc private func backButtonTapped() {
        navigationController?.popViewController(animated: true)
    }
    
    // Helper function to preprocess image for better emotion detection
    private func preprocessImageBuffer(_ buffer: CVPixelBuffer) -> CVPixelBuffer? {
        // Create a CIImage from the buffer
        let ciImage = CIImage(cvPixelBuffer: buffer)
        
        // 1. Convert to grayscale since model was trained on grayscale images
        let grayscaleFilter = CIFilter(name: "CIColorControls")
        grayscaleFilter?.setValue(ciImage, forKey: kCIInputImageKey)
        grayscaleFilter?.setValue(0, forKey: kCIInputSaturationKey) // 0 = grayscale
        
        guard let outputImage = grayscaleFilter?.outputImage else {
            return nil
        }
        
        // 2. Create a CGImage from the filtered CIImage
        let context = CIContext()
        guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
            return nil
        }
        
        // 3. Create a new pixel buffer with EXACT dimensions that model expects (299x299)
        return pixelBufferFromCGImage(cgImage, size: CGSize(width: 299, height: 299))
    }
    
    // Create a pixel buffer from CGImage
    private func pixelBufferFromCGImage(_ image: CGImage, size: CGSize) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let options = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ] as CFDictionary
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            options,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        if let context = context {
            context.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        }
        
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        
        return buffer
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension LiveFeedViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Process every few frames to reduce CPU usage
        frameCounter += 1
        if frameCounter % 3 != 0 {
            return
        }
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Process the frame for face detection
        detectFaces(in: imageBuffer)
    }
}
