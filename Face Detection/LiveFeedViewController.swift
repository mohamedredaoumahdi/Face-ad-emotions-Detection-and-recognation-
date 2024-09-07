import AVFoundation //AVFoundation for handling audio and video data
import UIKit //UIKit for UI components
import Vision //Vision for computer vision tasks
import CoreML

class LiveFeedViewController: UIViewController {
    //Manages the flow of data from the camera
    private let captureSession = AVCaptureSession()
    //Provides a live preview of the camera feed
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    //Outputs video frames as sample buffers.
    private let videoDataOutput = AVCaptureVideoDataOutput()
    //Array to store CAShapeLayer objects for face rectangles and landmarks.
    private var faceLayers: [CAShapeLayer] = []
    //SetUp our emotionLabel
    private let emotionLabel: UILabel = {
        let label = UILabel()
        label.textColor = UIColor.blue
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 18)
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    //Initializes the camera setup and starts the capture session
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupEmotionLabel()
        captureSession.startRunning()
    }
    
    // Setup the position of our emotionLabel
    
    private func setupEmotionLabel() {
        self.view.addSubview(emotionLabel)
        NSLayoutConstraint.activate([
            emotionLabel.topAnchor.constraint(equalTo: view.topAnchor, constant: 20),
            emotionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            emotionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            emotionLabel.heightAnchor.constraint(equalToConstant: 30)
        ])
    }
    //Adjusts the preview layer frame to match the view's frame.
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.view.frame
    }
    //Configures the camera input for the capture session.
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
    //Sets up the preview layer and video data output.
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
}
//Conformance to the AVCaptureVideoDataOutputSampleBufferDelegate protocol to receive video frames.
extension LiveFeedViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    //The captureOutput(_:didOutput:from:) method handles each captured video frame.
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
          return
        }
        
        let faceDetectionQueue = DispatchQueue(label: "faceDetectionQueue")
        let emotionDetectionQueue = DispatchQueue(label: "emotionDetectionQueue", qos: .userInitiated)

        // face detection code
        
        faceDetectionQueue.async{
            let faceDetectionRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request: VNRequest, error: Error?) in
                DispatchQueue.main.async {
                    self.faceLayers.forEach({ drawing in drawing.removeFromSuperlayer() })

                    if let observations = request.results as? [VNFaceObservation] {
                        self.handleFaceDetectionObservations(observations: observations)
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
        
        // Emotion detection code
        
        emotionDetectionQueue.async{
            if let emotion = self.detectEmotion(for: imageBuffer) {
                self.updateEmotionLabel(emotion)
            }
        }
        
    }
    //Face Detection:
    
    //Takes an array of face observations and draws yellow rectangles around detected faces.
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
            
            //FACE LANDMARKS
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
    //Draws green lines around individual facial landmarks (e.g., eyes, eyebrows, nose, lips).
    private func handleLandmark(_ eye: VNFaceLandmarkRegion2D, faceBoundingBox: CGRect) {
        let landmarkPath = CGMutablePath()
        let landmarkPathPoints = eye.normalizedPoints
            .map({ eyePoint in
                CGPoint(
                    x: eyePoint.y * faceBoundingBox.height + faceBoundingBox.origin.x,
                    y: eyePoint.x * faceBoundingBox.width + faceBoundingBox.origin.y)
            })
        //Constructs a path for facial landmarks using normalized points and transforms them to the preview layer coordinates.
        landmarkPath.addLines(between: landmarkPathPoints)
        landmarkPath.closeSubpath()
        let landmarkLayer = CAShapeLayer()
        landmarkLayer.path = landmarkPath
        landmarkLayer.fillColor = UIColor.clear.cgColor
        landmarkLayer.strokeColor = UIColor.green.cgColor
        //Adds the created shape layers (yellow rectangles for faces and green lines for landmarks) to the view.
        self.faceLayers.append(landmarkLayer)
        self.view.layer.addSublayer(landmarkLayer)
    }
    
    private func detectEmotion(for imageBuffer: CVPixelBuffer) -> String? {
        do {
            // Load the Core ML model
            let emotionModel =  EmotionClassificationModel()

            // Prepare the input features
            let emotionModelInput = EmotionClassificationModelInput(image: imageBuffer)

            // Perform prediction
            let emotionModelOutput = try emotionModel.prediction(input: emotionModelInput)

            // Access the predicted emotion
            let predictedEmotion = emotionModelOutput.classLabel

            return predictedEmotion
        } catch let error {
            print("Error loading or using the Core ML model: \(error)")
            return nil
        }
    }
    private func updateEmotionLabel(_ emotion: String) {
        DispatchQueue.main.async {
            self.emotionLabel.text = "Emotion: \(emotion)"
        }
    }
}
