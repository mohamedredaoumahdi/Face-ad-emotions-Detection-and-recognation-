

import UIKit
import Vision
import CoreML

class StillImageViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    var scaledImageRect: CGRect?
    var imageName : String = "neutral"
    @IBOutlet weak var emotionLabel: UILabel!
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        emotionLabel.font = UIFont.systemFont(ofSize: 24, weight: .bold)
        emotionLabel.textColor = .blue
        emotionLabel.textAlignment = .center
        emotionLabel.text = "Emotion: Loading..."
        
        if let image = UIImage(named: imageName) {
            imageView.image = image
            
            guard let cgImage = image.cgImage else {
                return
            }
    
            calculateScaledImageRect()
            performVisionRequest(image: cgImage)
        }
        
    }
    func predict(with imageName: String) -> EmotionClassificationModelOutput? {
        do {
            // Load the Core ML model
            let config = MLModelConfiguration()
            let model = try EmotionClassificationModel(configuration: config)
            
            // Load the image from the bundle
            guard let image = UIImage(named: imageName) else {
                print("Error: Unable to load image.")
                return nil
            }

            // Convert the UIImage to a CVPixelBuffer
            guard let pixelBuffer = pixelBuffer(from: image) else {
                print("Error: Unable to convert image to pixel buffer.")
                return nil
            }

            // Make a prediction using the model
            let prediction = try model.prediction(image: pixelBuffer)
            return prediction
        } catch {
            print("Error loading the model: \(error)")
            assertionFailure(error.localizedDescription)
            return nil
        }
    }
    func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        let size = CGSize(width: 48, height: 48) // Adjust the size based on your model input requirements

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height), kCVPixelFormatType_32BGRA, nil, &pixelBuffer)

        guard status == kCVReturnSuccess else {
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
         
         let faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: self.handleFaceDetectionRequest)

         let requests = [faceDetectionRequest]
         let imageRequestHandler = VNImageRequestHandler(cgImage: image,
                                                         orientation: .up,
                                                         options: [:])
         
         DispatchQueue.global(qos: .userInitiated).async {
             do {
                 try imageRequestHandler.perform(requests)
             } catch let error as NSError {
                 print(error)
                 return
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
