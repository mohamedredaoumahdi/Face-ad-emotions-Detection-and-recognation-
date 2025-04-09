import AVFoundation
import UIKit
import Vision
import CoreML

class LiveFeedViewController: UIViewController {
    // Manages the flow of data from the camera
    private let captureSession = AVCaptureSession()
    // Provides a live preview of the camera feed
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    // Outputs video frames as sample buffers
    private let videoDataOutput = AVCaptureVideoDataOutput()
    // Array to store CAShapeLayer objects for face rectangles and landmarks
    private var faceLayers: [CAShapeLayer] = []
    // Frame counter for processing optimization
    private var frameCounter = 0
    // SetUp our emotionLabel
    private let emotionLabel: UILabel = {
        let label = UILabel()
        label.textColor = UIColor.blue
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 18)
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    // Debug info label
    private let debugLabel: UILabel = {
        let label = UILabel()
        label.textColor = UIColor.white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        label.textAlignment = .left
        label.font = UIFont.systemFont(ofSize: 12)
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupEmotionLabel()
        setupDebugLabel()
        
        // Start capture session on a background thread
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    private func setupEmotionLabel() {
        self.view.addSubview(emotionLabel)
        NSLayoutConstraint.activate([
            emotionLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            emotionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            emotionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            emotionLabel.heightAnchor.constraint(equalToConstant: 30)
        ])
    }
    
    private func setupDebugLabel() {
        self.view.addSubview(debugLabel)
        NSLayoutConstraint.activate([
            debugLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            debugLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10),
            debugLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -10),
            debugLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 60)
        ])
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.view.frame
    }
    
    private func setupCamera() {
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .front)
        if let device = deviceDiscoverySession.devices.first {
            if let deviceInput = try? AVCaptureDeviceInput(device: device) {
                if captureSession.canAddInput(deviceInput) {
                    captureSession.addInput(deviceInput)
                    setupPreview()
                }
            }
        }
    }
    
    private func setupPreview() {
        self.previewLayer.videoGravity = .resizeAspectFill
        self.view.layer.addSublayer(self.previewLayer)
        self.previewLayer.frame = self.view.frame
        
        self.videoDataOutput.videoSettings = [(kCVPixelBufferPixelFormatTypeKey as NSString) : NSNumber(value: kCVPixelFormatType_32BGRA)] as [String : Any]
        
        self.videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera queue"))
        self.captureSession.addOutput(self.videoDataOutput)
        
        let videoConnection = self.videoDataOutput.connection(with: .video)
        videoConnection?.videoOrientation = .portrait
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

extension LiveFeedViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // Process every 5th frame to reduce CPU usage
        frameCounter += 1
        if frameCounter % 5 != 0 {
            return
        }
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let faceDetectionQueue = DispatchQueue(label: "faceDetectionQueue")
        let emotionDetectionQueue = DispatchQueue(label: "emotionDetectionQueue", qos: .userInitiated)
        
        // Face detection code
        faceDetectionQueue.async {
            let faceDetectionRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request: VNRequest, error: Error?) in
                DispatchQueue.main.async {
                    self.faceLayers.forEach({ drawing in drawing.removeFromSuperlayer() })
                    
                    if let observations = request.results as? [VNFaceObservation] {
                        self.handleFaceDetectionObservations(observations: observations)
                        
                        // Update debug info
                        if let firstFace = observations.first {
                            let faceSize = "Face size: \(firstFace.boundingBox.width) x \(firstFace.boundingBox.height)"
                            DispatchQueue.main.async {
                                self.debugLabel.text = faceSize
                            }
                        }
                    }
                }
            })
            
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: imageBuffer, orientation: .leftMirrored, options: [:])
            
            do {
                try imageRequestHandler.perform([faceDetectionRequest])
            } catch {
                print(error.localizedDescription)
            }
        }
        
        // Emotion detection code with preprocessing
        emotionDetectionQueue.async {
            // Try to preprocess the image to improve emotion detection
            let preprocessedBuffer = self.preprocessImageBuffer(imageBuffer)
            if let emotion = self.detectEmotion(for: preprocessedBuffer ?? imageBuffer) {
                self.updateEmotionLabel(emotion)
            }
        }
    }
    
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
    
    private func handleFaceDetectionObservations(observations: [VNFaceObservation]) {
        for observation in observations {
            let faceRectConverted = self.previewLayer.layerRectConverted(fromMetadataOutputRect: observation.boundingBox)
            let faceRectanglePath = CGPath(rect: faceRectConverted, transform: nil)
            
            let faceLayer = CAShapeLayer()
            faceLayer.path = faceRectanglePath
            faceLayer.fillColor = UIColor.clear.cgColor
            faceLayer.strokeColor = UIColor.yellow.cgColor
            
            self.faceLayers.append(faceLayer)
            self.view.layer.addSublayer(faceLayer)
            
            // FACE LANDMARKS
            if let landmarks = observation.landmarks {
                if let leftEye = landmarks.leftEye {
                    self.handleLandmark(leftEye, faceBoundingBox: faceRectConverted)
                }
                if let leftEyebrow = landmarks.leftEyebrow {
                    self.handleLandmark(leftEyebrow, faceBoundingBox: faceRectConverted)
                }
                if let rightEye = landmarks.rightEye {
                    self.handleLandmark(rightEye, faceBoundingBox: faceRectConverted)
                }
                if let rightEyebrow = landmarks.rightEyebrow {
                    self.handleLandmark(rightEyebrow, faceBoundingBox: faceRectConverted)
                }
                
                if let nose = landmarks.nose {
                    self.handleLandmark(nose, faceBoundingBox: faceRectConverted)
                }
                
                if let outerLips = landmarks.outerLips {
                    self.handleLandmark(outerLips, faceBoundingBox: faceRectConverted)
                }
                if let innerLips = landmarks.innerLips {
                    self.handleLandmark(innerLips, faceBoundingBox: faceRectConverted)
                }
            }
        }
    }
    
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
    
    private func detectEmotion(for imageBuffer: CVPixelBuffer) -> String? {
        do {
            guard let modelURL = Bundle.main.url(forResource: "EmotionClassificationModel", withExtension: "mlmodelc") else {
                print("Error: Model file not found in bundle")
                return "Model not found"
            }
            
            // Configuration matching how the model was trained
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            
            // Get model description from the loaded model instead
            let modelDescription = model.modelDescription
            print("Model input: \(modelDescription.inputDescriptionsByName)")
            print("Model output: \(modelDescription.outputDescriptionsByName)")
            
            let emotionClassifier = try EmotionClassificationModel(model: model)
            let emotionModelInput = EmotionClassificationModelInput(image: imageBuffer)
            let emotionModelOutput = try emotionClassifier.prediction(input: emotionModelInput)
            
            // Print the emotion probabilities to see confidence levels
            let probabilities = emotionModelOutput.classLabelProbs
            let sortedProbs = probabilities.sorted { $0.value > $1.value }
            print("Emotion probabilities:")
            for (emotion, prob) in sortedProbs.prefix(3) {
                print("  \(emotion): \(prob * 100)%")
            }
            
            return emotionModelOutput.classLabel
        } catch {
            print("Emotion detection error: \(error)")
            return "Error"
        }
    }
    
    private func updateEmotionLabel(_ emotion: String) {
        DispatchQueue.main.async {
            self.emotionLabel.text = "Emotion: \(emotion)"
        }
    }
}
