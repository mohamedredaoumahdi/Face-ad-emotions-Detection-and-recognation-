

import UIKit
import Vision
import CoreML

class StillImageViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    var scaledImageRect: CGRect?
    var imageName : String = "neutral"
    @IBOutlet weak var emotionLabel: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
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
    func predict(with imageName: String) -> EmotionClassificationModelOutput? {
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
            
            // Load the image from the bundle
            guard let image = UIImage(named: imageName) else {
                print("Error: Unable to load image.")
                return nil
            }
            
            // Resize image before processing
            let processingSize = CGSize(width: 300, height: 300)
            guard let resizedImage = resizeImage(image, to: processingSize) else {
                print("Error: Unable to resize image.")
                return nil
            }
            
            // Convert the UIImage to a CVPixelBuffer
            guard let pixelBuffer = pixelBuffer(from: resizedImage) else {
                print("Error: Unable to convert image to pixel buffer.")
                return nil
            }
            
            // Make a prediction using the model
            let prediction = try emotionClassifier.prediction(image: pixelBuffer)
            return prediction
        } catch {
            print("Detailed error loading model: \(error)")
            return nil
        }
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
            
        let size = CGSize(width: 48, height: 48) // Model input requirements
            
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
                    print(observation.boundingBox)
                    
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
                DispatchQueue.main.async {
                    if let prediction = self.predict(with: self.imageName) {
                        print(prediction.classLabel)
                        self.emotionLabel.text = "Emotion: \(prediction.classLabel)"
                        self.view.setNeedsLayout()
                        self.view.layoutIfNeeded()
                    }
                }
                
            }
        }
    }
}
