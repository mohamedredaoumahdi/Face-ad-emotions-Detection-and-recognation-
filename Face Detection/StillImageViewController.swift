import UIKit
import Vision
import CoreML
import PhotosUI

class StillImageViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    var scaledImageRect: CGRect?
    var imageName: String = "neutral"
    @IBOutlet weak var emotionLabel: UILabel!
    
    // Debug label to show emotion probabilities
    private let debugLabel: UILabel = {
        let label = UILabel()
        label.textColor = UIColor.white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.textAlignment = .left
        label.font = UIFont.systemFont(ofSize: 12)
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Add a gallery button to the view
        let galleryButton = UIButton(type: .system)
        galleryButton.setTitle("Choose from Gallery", for: .normal)
        galleryButton.addTarget(self, action: #selector(openGallery), for: .touchUpInside)
        galleryButton.translatesAutoresizingMaskIntoConstraints = false
        
        // Add debug label
        self.view.addSubview(debugLabel)
        
        self.view.addSubview(galleryButton)
        
        // Position the button at the bottom of the screen
        NSLayoutConstraint.activate([
            galleryButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            galleryButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            galleryButton.widthAnchor.constraint(greaterThanOrEqualToConstant: 200),
            galleryButton.heightAnchor.constraint(equalToConstant: 44),
            
            // Position debug label
            debugLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 10),
            debugLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -10),
            debugLabel.bottomAnchor.constraint(equalTo: galleryButton.topAnchor, constant: -20),
            debugLabel.heightAnchor.constraint(greaterThanOrEqualToConstant: 100)
        ])
        
        // Check if model exists in bundle
        if let modelURL = Bundle.main.url(forResource: "EmotionClassificationModel", withExtension: "mlmodelc") {
            print("Model found at: \(modelURL)")
        } else {
            print("Model not found in bundle!")
            
            // Check all bundle resources
            let urls = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)
            print("Available model files: \(urls ?? [])")
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        emotionLabel.font = UIFont.systemFont(ofSize: 24, weight: .bold)
        emotionLabel.textColor = .blue
        emotionLabel.textAlignment = .center
        emotionLabel.text = "Emotion: Loading..."
        
        if let image = UIImage(named: imageName) {
            imageView.image = image
            
            // Resize image to smaller dimensions before processing
            let processingSize = CGSize(width: 300, height: 300)
            guard let resizedImage = resizeImage(image, to: processingSize),
                  let cgImage = resizedImage.cgImage else {
                emotionLabel.text = "Error: Image processing failed"
                return
            }
            
            calculateScaledImageRect()
            performVisionRequest(image: cgImage)
        }
    }
    
    @objc func openGallery() {
        if #available(iOS 14, *) {
            // Use PHPickerViewController for iOS 14+
            var configuration = PHPickerConfiguration()
            configuration.filter = .images
            configuration.selectionLimit = 1
            
            let picker = PHPickerViewController(configuration: configuration)
            picker.delegate = self
            present(picker, animated: true)
        } else {
            // Fallback to UIImagePickerController for older iOS
            let picker = UIImagePickerController()
            picker.sourceType = .photoLibrary
            picker.delegate = self
            present(picker, animated: true)
        }
    }
    
    // Process the selected image
    func processSelectedImage(_ image: UIImage) {
        // Display the selected image
        imageView.image = image
        emotionLabel.text = "Emotion: Loading..."
        
        // Resize image for processing
        let processingSize = CGSize(width: 300, height: 300)
        guard let resizedImage = resizeImage(image, to: processingSize),
              let cgImage = resizedImage.cgImage else {
            emotionLabel.text = "Error: Image processing failed"
            return
        }
        
        calculateScaledImageRect()
        performVisionRequest(image: cgImage)
    }
    
    func predict(with image: UIImage) -> EmotionClassificationModelOutput? {
        do {
            // Load model from main bundle explicitly
            guard let modelURL = Bundle.main.url(forResource: "EmotionClassificationModel", withExtension: "mlmodelc") else {
                print("Error: Model file not found in bundle")
                return nil
            }
            
            print("Loading model from: \(modelURL)")
            
            // Create configuration with explicit resource constraints
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU // Try using CPU and GPU together
            config.allowLowPrecisionAccumulationOnGPU = true // Better performance
            
            // Load model directly from URL
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let emotionClassifier = try EmotionClassificationModel(model: model)
            
            // First convert the image to grayscale
            let grayscaleImage = convertToGrayscale(image)
            
            // Resize to exact dimensions expected by model (299x299)
            let modelSize = CGSize(width: 299, height: 299)
            guard let resizedForModel = resizeImage(grayscaleImage, to: modelSize) else {
                print("Error: Unable to resize image to model dimensions.")
                return nil
            }
            
            // Convert the UIImage to a CVPixelBuffer
            guard let pixelBuffer = pixelBuffer(from: resizedForModel) else {
                print("Error: Unable to convert image to pixel buffer.")
                return nil
            }
            
            // Make a prediction using the model
            let prediction = try emotionClassifier.prediction(image: pixelBuffer)
            
            // Get probabilities and print top 3
            let probabilities = prediction.classLabelProbs
            let sortedProbs = probabilities.sorted { $0.value > $1.value }
            print("Emotion probabilities:")
            for (emotion, prob) in sortedProbs.prefix(3) {
                print("  \(emotion): \(prob * 100)%")
            }
            
            // Update debug label with probabilities
            DispatchQueue.main.async {
                var debugText = "Emotion Probabilities:\n"
                for (emotion, prob) in sortedProbs.prefix(3) {
                    debugText += "- \(emotion): \(Int(prob * 100))%\n"
                }
                self.debugLabel.text = debugText
            }
            
            return prediction
        } catch {
            print("Detailed error loading model: \(error)")
            return nil
        }
    }
    
    // Convert image to grayscale
    func convertToGrayscale(_ image: UIImage) -> UIImage {
        let context = CIContext(options: nil)
        if let filter = CIFilter(name: "CIPhotoEffectMono") {
            filter.setValue(CIImage(image: image)!, forKey: kCIInputImageKey)
            if let output = filter.outputImage,
               let cgImage = context.createCGImage(output, from: output.extent) {
                return UIImage(cgImage: cgImage)
            }
        }
        return image // Return original if conversion fails
    }

    // Helper function for image resizing
    func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        print("Converting image: \(image.size.width) x \(image.size.height)")
        
        // Use the size the model expects (299x299)
        let size = image.size
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32BGRA,
            [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary,
            &pixelBuffer
        )
        
        if status != kCVReturnSuccess {
            print("Failed to create pixel buffer with status: \(status)")
            return nil
        }

        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.translateBy(x: 0, y: size.height)
        context?.scaleBy(x: 1, y: -1)

        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsPopContext()

        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

        return pixelBuffer
    }
    
    private func calculateScaledImageRect() {
        guard let image = imageView.image else {
            return
        }

        guard let cgImage = image.cgImage else {
            return
        }

        let originalWidth = CGFloat(cgImage.width)
        let originalHeight = CGFloat(cgImage.height)

        let imageFrame = imageView.frame
        let widthRatio = originalWidth / imageFrame.width
        let heightRatio = originalHeight / imageFrame.height

        // ScaleAspectFit
        let scaleRatio = max(widthRatio, heightRatio)

        let scaledImageWidth = originalWidth / scaleRatio
        let scaledImageHeight = originalHeight / scaleRatio

        let scaledImageX = (imageFrame.width - scaledImageWidth) / 2
        let scaledImageY = (imageFrame.height - scaledImageHeight) / 2
        
        self.scaledImageRect = CGRect(x: scaledImageX, y: scaledImageY, width: scaledImageWidth, height: scaledImageHeight)
    }
    
    private func performVisionRequest(image: CGImage) {
        // Create a new request with better error handling
        let faceDetectionRequest = VNDetectFaceRectanglesRequest { [weak self] request, error in
            if let error = error {
                print("Face detection error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self?.emotionLabel.text = "Error: Face detection failed"
                }
                return
            }
            
            self?.handleFaceDetectionRequest(request: request, error: error)
        }
        
        // Increase request priority and add configuration
        faceDetectionRequest.usesCPUOnly = false // Use Neural Engine if available
        
        let requests = [faceDetectionRequest]
        
        // Explicitly specify orientation and other parameters
        let imageRequestHandler = VNImageRequestHandler(
            cgImage: image,
            orientation: .up,
            options: [VNImageOption.ciContext: CIContext()]
        )
        
        // Use higher priority queue
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try imageRequestHandler.perform(requests)
            } catch let error as NSError {
                print("ImageRequestHandler error: \(error), \(error.userInfo)")
                DispatchQueue.main.async {
                    self.emotionLabel.text = "Error: Vision processing failed"
                }
            }
        }
    }
    
    private func handleFaceDetectionRequest(request: VNRequest?, error: Error?) {
        if let requestError = error as NSError? {
            print(requestError)
            return
        }
        
        guard let imageRect = self.scaledImageRect else {
            return
        }
            
        let imageWidth = imageRect.size.width
        let imageHeight = imageRect.size.height
        
        DispatchQueue.main.async {
            
            self.imageView.layer.sublayers = nil
            if let results = request?.results as? [VNFaceObservation] {
                
                for observation in results {
                    print("Detected face at \(observation.boundingBox)")
                    
                    var scaledObservationRect = observation.boundingBox
                    scaledObservationRect.origin.x = imageRect.origin.x + (observation.boundingBox.origin.x * imageWidth)
                    scaledObservationRect.origin.y = imageRect.origin.y + (1 - observation.boundingBox.origin.y - observation.boundingBox.height) * imageHeight
                    scaledObservationRect.size.width *= imageWidth
                    scaledObservationRect.size.height *= imageHeight
                    
                    let faceRectanglePath = CGPath(rect: scaledObservationRect, transform: nil)
                    
                    let faceLayer = CAShapeLayer()
                    faceLayer.path = faceRectanglePath
                    faceLayer.fillColor = UIColor.clear.cgColor
                    faceLayer.strokeColor = UIColor.yellow.cgColor
                    self.imageView.layer.addSublayer(faceLayer)
                }
                
                // Only process if we found at least one face
                if !results.isEmpty {
                    // Now predict using the actual image from imageView
                    if let currentImage = self.imageView.image {
                        if let prediction = self.predict(with: currentImage) {
                            // Apply a confidence threshold - only show emotion if confidence is high enough
                            let topEmotion = prediction.classLabel
                            let confidence = prediction.classLabelProbs[topEmotion] ?? 0
                            
                            if confidence > 0.6 {
                                self.emotionLabel.text = "Emotion: \(topEmotion) (\(Int(confidence * 100))%)"
                            } else {
                                self.emotionLabel.text = "Emotion: Uncertain"
                            }
                            
                            self.view.setNeedsLayout()
                            self.view.layoutIfNeeded()
                        }
                    }
                } else {
                    self.emotionLabel.text = "No face detected"
                }
            }
        }
    }
}

// MARK: - PHPicker Delegate (iOS 14+)
@available(iOS 14, *)
extension StillImageViewController: PHPickerViewControllerDelegate {
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        picker.dismiss(animated: true)
        
        guard let result = results.first else { return }
        
        result.itemProvider.loadObject(ofClass: UIImage.self) { [weak self] object, error in
            if let image = object as? UIImage {
                DispatchQueue.main.async {
                    self?.processSelectedImage(image)
                }
            }
        }
    }
}

// MARK: - UIImagePicker Delegate (iOS 13 and below)
extension StillImageViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true)
        
        if let image = info[.originalImage] as? UIImage {
            processSelectedImage(image)
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
    }
}
