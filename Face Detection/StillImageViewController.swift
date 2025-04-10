import UIKit
import Vision
import CoreML
import PhotosUI

class StillImageViewController: UIViewController {
    
    // MARK: - Properties
    
    private var scaledImageRect: CGRect?
    private var selectedImage: UIImage?
    private var currentPrediction: EmotionClassificationModelOutput?
    
    // MARK: - UI Components
    
    private let imageView: UIImageView = {
        let imageView = UIImageView()
        imageView.contentMode = .scaleAspectFit
        imageView.clipsToBounds = true
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.backgroundColor = .black
        return imageView
    }()
    
    private let emotionLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 20, weight: .bold)
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.layer.cornerRadius = 12
        label.layer.masksToBounds = true
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let galleryButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle(" Import", for: .normal)
        button.setImage(UIImage(systemName: "photo.on.rectangle"), for: .normal)
        button.tintColor = .white
        button.backgroundColor = UIColor(named: "AccentColor") ?? .systemPurple
        button.layer.cornerRadius = 16
        button.titleLabel?.font = UIFont.systemFont(ofSize: 17, weight: .semibold)
        button.contentEdgeInsets = UIEdgeInsets(top: 12, left: 16, bottom: 12, right: 16)
        button.imageEdgeInsets = UIEdgeInsets(top: 0, left: 0, bottom: 0, right: 8)
        button.translatesAutoresizingMaskIntoConstraints = false
        
        // Add shadow
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 4)
        button.layer.shadowRadius = 8
        button.layer.shadowOpacity = 0.2
        return button
    }()
    
    private let cameraButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle(" Camera", for: .normal)
        button.setImage(UIImage(systemName: "camera"), for: .normal)
        button.tintColor = .white
        button.backgroundColor = UIColor(named: "AccentColor") ?? .systemPurple
        button.layer.cornerRadius = 16
        button.titleLabel?.font = UIFont.systemFont(ofSize: 17, weight: .semibold)
        button.contentEdgeInsets = UIEdgeInsets(top: 12, left: 16, bottom: 12, right: 16)
        button.imageEdgeInsets = UIEdgeInsets(top: 0, left: 0, bottom: 0, right: 8)
        button.translatesAutoresizingMaskIntoConstraints = false
        
        // Add shadow
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 4)
        button.layer.shadowRadius = 8
        button.layer.shadowOpacity = 0.2
        return button
    }()
    
    private let emotionDetailsView: UIView = {
        let view = UIView()
        view.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        view.layer.cornerRadius = 16
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }()
    
    private let emotionBarStackView: UIStackView = {
        let stackView = UIStackView()
        stackView.axis = .vertical
        stackView.spacing = 8
        stackView.distribution = .fillEqually
        stackView.translatesAutoresizingMaskIntoConstraints = false
        return stackView
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
    
    private let placeholderView: UIView = {
        let view = UIView()
        view.backgroundColor = UIColor.darkGray.withAlphaComponent(0.3)
        view.translatesAutoresizingMaskIntoConstraints = false
        // Ensure placeholder doesn't block touches to buttons underneath
        view.isUserInteractionEnabled = false
        return view
    }()
    
    private let placeholderImageView: UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(systemName: "photo.on.rectangle.angled")
        imageView.contentMode = .scaleAspectFit
        imageView.tintColor = .white.withAlphaComponent(0.7)
        imageView.translatesAutoresizingMaskIntoConstraints = false
        return imageView
    }()
    
    private let placeholderLabel: UILabel = {
        let label = UILabel()
        label.text = "Select an image to analyze"
        label.textColor = .white
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 18, weight: .medium)
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    // MARK: - Lifecycle Methods
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupPlaceholderView()
        setupActions()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        // Show placeholder instead of loading a default image
        showPlaceholderView(true)
        emotionLabel.text = "Select an image to analyze"
        emotionDetailsView.isHidden = true
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        if let image = imageView.image, imageView.bounds.size != .zero {
            calculateScaledImageRect()
        }
    }
    
    // MARK: - Setup Methods
    
    private func setupUI() {
        navigationController?.setNavigationBarHidden(true, animated: false)
        view.backgroundColor = UIColor(named: "BackgroundColor") ?? .systemBackground
        
        // Set up image view - make sure it doesn't block interaction
        view.addSubview(imageView)
        imageView.isUserInteractionEnabled = false
        
        // Add UI elements - these should be on top of the view hierarchy
        view.addSubview(emotionLabel)
        view.addSubview(emotionDetailsView)
        emotionDetailsView.addSubview(emotionBarStackView)
        view.addSubview(galleryButton)
        view.addSubview(cameraButton)
        view.addSubview(backButton)
        
        // Ensure the buttons have user interaction enabled
        galleryButton.isUserInteractionEnabled = true
        cameraButton.isUserInteractionEnabled = true
        backButton.isUserInteractionEnabled = true
        
        // Make sure buttons are fully visible and not transparent
        galleryButton.alpha = 1.0
        cameraButton.alpha = 1.0
        
        // Set constraints
        NSLayoutConstraint.activate([
            // Image view takes full screen
            imageView.topAnchor.constraint(equalTo: view.topAnchor),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            imageView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Emotion label at top
            emotionLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            emotionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            emotionLabel.widthAnchor.constraint(lessThanOrEqualTo: view.widthAnchor, constant: -40),
            emotionLabel.heightAnchor.constraint(equalToConstant: 40),
            
            // Gallery button
            galleryButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            galleryButton.trailingAnchor.constraint(equalTo: view.centerXAnchor, constant: -10),
            galleryButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Camera button
            cameraButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            cameraButton.leadingAnchor.constraint(equalTo: view.centerXAnchor, constant: 10),
            cameraButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Emotion details view
            emotionDetailsView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            emotionDetailsView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            emotionDetailsView.bottomAnchor.constraint(equalTo: galleryButton.topAnchor, constant: -20),
            emotionDetailsView.heightAnchor.constraint(equalToConstant: 200),
            
            // Emotion bar stack view
            emotionBarStackView.topAnchor.constraint(equalTo: emotionDetailsView.topAnchor, constant: 16),
            emotionBarStackView.leadingAnchor.constraint(equalTo: emotionDetailsView.leadingAnchor, constant: 16),
            emotionBarStackView.trailingAnchor.constraint(equalTo: emotionDetailsView.trailingAnchor, constant: -16),
            emotionBarStackView.bottomAnchor.constraint(equalTo: emotionDetailsView.bottomAnchor, constant: -16),
            
            // Back button
            backButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            backButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            backButton.widthAnchor.constraint(equalToConstant: 40),
            backButton.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        // Set initial state
        emotionLabel.text = "Select an image to analyze"
        emotionDetailsView.isHidden = true
    }
    
    private func setupPlaceholderView() {
        // Insert placeholder view BEFORE the buttons so they remain on top and clickable
        view.insertSubview(placeholderView, belowSubview: backButton)
        placeholderView.addSubview(placeholderImageView)
        placeholderView.addSubview(placeholderLabel)
        
        // Make sure placeholder doesn't interfere with user interaction
        placeholderView.isUserInteractionEnabled = false
        
        NSLayoutConstraint.activate([
            placeholderView.topAnchor.constraint(equalTo: view.topAnchor),
            placeholderView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            placeholderView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            placeholderView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            placeholderImageView.centerXAnchor.constraint(equalTo: placeholderView.centerXAnchor),
            placeholderImageView.centerYAnchor.constraint(equalTo: placeholderView.centerYAnchor, constant: -50),
            placeholderImageView.widthAnchor.constraint(equalToConstant: 100),
            placeholderImageView.heightAnchor.constraint(equalToConstant: 100),
            
            placeholderLabel.topAnchor.constraint(equalTo: placeholderImageView.bottomAnchor, constant: 20),
            placeholderLabel.centerXAnchor.constraint(equalTo: placeholderView.centerXAnchor),
            placeholderLabel.leadingAnchor.constraint(equalTo: placeholderView.leadingAnchor, constant: 20),
            placeholderLabel.trailingAnchor.constraint(equalTo: placeholderView.trailingAnchor, constant: -20)
        ])
    }
    
    private func setupActions() {
        galleryButton.addTarget(self, action: #selector(openGallery), for: .touchUpInside)
        cameraButton.addTarget(self, action: #selector(openCamera), for: .touchUpInside)
    }
    
    // MARK: - Action Methods
    
    @objc private func openGallery() {
        if #available(iOS 14, *) {
            var configuration = PHPickerConfiguration()
            configuration.filter = .images
            configuration.selectionLimit = 1
            
            let picker = PHPickerViewController(configuration: configuration)
            picker.delegate = self
            present(picker, animated: true)
        } else {
            let picker = UIImagePickerController()
            picker.sourceType = .photoLibrary
            picker.delegate = self
            present(picker, animated: true)
        }
    }
    
    @objc private func openCamera() {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = self
        present(picker, animated: true)
    }
    
    @objc private func backButtonTapped() {
        navigationController?.popViewController(animated: true)
    }
    
    // MARK: - Helper Methods
    
    private func showPlaceholderView(_ show: Bool) {
        placeholderView.isHidden = !show
        imageView.isHidden = show
        
        // Make sure buttons are always on top and interactive
        if show {
            // Bring buttons to front when placeholder is shown
            view.bringSubviewToFront(galleryButton)
            view.bringSubviewToFront(cameraButton)
            view.bringSubviewToFront(backButton)
            view.bringSubviewToFront(emotionLabel)
        }
    }
    
    // MARK: - Image Processing
    
    private func processSelectedImage(_ image: UIImage) {
        // Hide placeholder and show the real image
        showPlaceholderView(false)
        
        // Don't set the image again, it's already set in the picker completion
        // Instead, just start the analysis
        emotionLabel.text = "Analyzing..."
        emotionDetailsView.isHidden = true
        
        // Clear any existing face layers
        imageView.layer.sublayers?.removeAll(where: { $0 is CAShapeLayer })
        
        // Calculate the scaled image rect for proper face overlay
        calculateScaledImageRect()
        
        // Process the image in background
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self, let cgImage = image.cgImage else { return }
            
            // Perform face detection
            self.performVisionRequest(image: cgImage)
            
            // Make emotion prediction
            if let prediction = self.predict(with: image) {
                self.currentPrediction = prediction
                
                DispatchQueue.main.async {
                    self.updateUIWithPrediction(prediction)
                }
            }
        }
    }
    
    private func calculateScaledImageRect() {
        // Wait until the view is laid out properly
        DispatchQueue.main.async { [weak self] in
            guard let self = self,
                  let image = self.imageView.image,
                  let cgImage = image.cgImage else { return }
            
            // Get the actual displayed size of the image in the imageView
            let imageFrame = self.imageView.bounds
            let imageViewAspectRatio = imageFrame.width / imageFrame.height
            
            let originalWidth = CGFloat(cgImage.width)
            let originalHeight = CGFloat(cgImage.height)
            let imageAspectRatio = originalWidth / originalHeight
            
            var scaledImageWidth: CGFloat
            var scaledImageHeight: CGFloat
            
            // Calculate displayed image size (accounting for aspectFit)
            if imageAspectRatio > imageViewAspectRatio {
                // Image is wider than view
                scaledImageWidth = imageFrame.width
                scaledImageHeight = scaledImageWidth / imageAspectRatio
            } else {
                // Image is taller than view
                scaledImageHeight = imageFrame.height
                scaledImageWidth = scaledImageHeight * imageAspectRatio
            }
            
            let scaledImageX = (imageFrame.width - scaledImageWidth) / 2
            let scaledImageY = (imageFrame.height - scaledImageHeight) / 2
            
            self.scaledImageRect = CGRect(
                x: scaledImageX,
                y: scaledImageY,
                width: scaledImageWidth,
                height: scaledImageHeight
            )
            
            print("Image frame: \(imageFrame), Scaled rect: \(String(describing: self.scaledImageRect))")
        }
    }
    
    private func performVisionRequest(image: CGImage) {
        let faceDetectionRequest = VNDetectFaceRectanglesRequest { [weak self] request, error in
            if let error = error {
                print("Face detection error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self?.emotionLabel.text = "Error: Face detection failed"
                }
                return
            }
            
            self?.handleFaceDetectionRequest(request: request)
        }
        
        // Use Neural Engine if available
        faceDetectionRequest.usesCPUOnly = false
        
        let handler = VNImageRequestHandler(
            cgImage: image,
            orientation: .up,
            options: [VNImageOption.ciContext: CIContext()]
        )
        
        do {
            try handler.perform([faceDetectionRequest])
        } catch {
            print("Vision request failed: \(error.localizedDescription)")
            DispatchQueue.main.async { [weak self] in
                self?.emotionLabel.text = "Error: Vision processing failed"
            }
        }
    }
    
    private func handleFaceDetectionRequest(request: VNRequest?) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            // Clear previous face layers
            self.imageView.layer.sublayers?.removeAll(where: { $0 is CAShapeLayer })
            
            guard let results = request?.results as? [VNFaceObservation], !results.isEmpty else {
                self.emotionLabel.text = "No face detected"
                return
            }
            
            guard let imageRect = self.scaledImageRect else { return }
            
            // Draw faces
            for observation in results {
                // Calculate face rectangle in view coordinates
                var faceRect = observation.boundingBox
                faceRect.origin.y = 1 - faceRect.origin.y - faceRect.height // Flip y-coordinate
                
                let scaledRect = CGRect(
                    x: imageRect.origin.x + (faceRect.origin.x * imageRect.width),
                    y: imageRect.origin.y + (faceRect.origin.y * imageRect.height),
                    width: faceRect.width * imageRect.width,
                    height: faceRect.height * imageRect.height
                )
                
                // Create face layer with rounded corners
                let faceLayer = CAShapeLayer()
                faceLayer.path = UIBezierPath(roundedRect: scaledRect, cornerRadius: 10).cgPath
                faceLayer.fillColor = UIColor.clear.cgColor
                faceLayer.strokeColor = UIColor.yellow.cgColor
                faceLayer.lineWidth = 3
                
                self.imageView.layer.addSublayer(faceLayer)
            }
        }
    }
    
    private func updateUIWithPrediction(_ prediction: EmotionClassificationModelOutput) {
        // Get probabilities
        let probabilities = prediction.classLabelProbs
        let sortedProbs = probabilities.sorted { $0.value > $1.value }
        
        // Update emotion label
        let topEmotion = prediction.classLabel
        let confidence = probabilities[topEmotion] ?? 0
        let confidencePercentage = Int(confidence * 100)
        
        if confidence > 0.5 {
            emotionLabel.text = "\(topEmotion) (\(confidencePercentage)%)"
            emotionLabel.backgroundColor = colorForEmotion(topEmotion).withAlphaComponent(0.7)
        } else {
            emotionLabel.text = "Uncertain emotion"
            emotionLabel.backgroundColor = UIColor.darkGray.withAlphaComponent(0.7)
        }
        
        // Show emotion details panel
        emotionDetailsView.isHidden = false
        
        // Clear existing bars
        emotionBarStackView.arrangedSubviews.forEach { $0.removeFromSuperview() }
        
        // Add emotion bars for top emotions
        for (emotion, probability) in sortedProbs.prefix(5) {
            let barView = createEmotionBarView(emotion: emotion, probability: probability)
            emotionBarStackView.addArrangedSubview(barView)
        }
    }
    
    private func createEmotionBarView(emotion: String, probability: Double) -> UIView {
        let containerView = UIView()
        
        // Label for emotion name
        let nameLabel = UILabel()
        nameLabel.text = emotion
        nameLabel.textColor = .white
        nameLabel.font = UIFont.systemFont(ofSize: 14)
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        
        // Progress view
        let progressView = UIProgressView(progressViewStyle: .bar)
        progressView.progress = Float(probability)
        progressView.progressTintColor = colorForEmotion(emotion)
        progressView.trackTintColor = UIColor.lightGray.withAlphaComponent(0.3)
        progressView.layer.cornerRadius = 2
        progressView.clipsToBounds = true
        progressView.translatesAutoresizingMaskIntoConstraints = false
        
        // Percentage label
        let percentageLabel = UILabel()
        percentageLabel.text = "\(Int(probability * 100))%"
        percentageLabel.textColor = .white
        percentageLabel.font = UIFont.systemFont(ofSize: 14)
        percentageLabel.translatesAutoresizingMaskIntoConstraints = false
        
        containerView.addSubview(nameLabel)
        containerView.addSubview(progressView)
        containerView.addSubview(percentageLabel)
        
        NSLayoutConstraint.activate([
            nameLabel.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            nameLabel.centerYAnchor.constraint(equalTo: containerView.centerYAnchor),
            nameLabel.widthAnchor.constraint(equalToConstant: 90),
            
            progressView.leadingAnchor.constraint(equalTo: nameLabel.trailingAnchor, constant: 8),
            progressView.centerYAnchor.constraint(equalTo: containerView.centerYAnchor),
            progressView.heightAnchor.constraint(equalToConstant: 8),
            
            percentageLabel.leadingAnchor.constraint(equalTo: progressView.trailingAnchor, constant: 8),
            percentageLabel.centerYAnchor.constraint(equalTo: containerView.centerYAnchor),
            percentageLabel.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            
            progressView.widthAnchor.constraint(equalTo: containerView.widthAnchor, multiplier: 0.5)
        ])
        
        return containerView
    }
    
    // MARK: - Emotion Classification
    
    private func predict(with image: UIImage) -> EmotionClassificationModelOutput? {
        do {
            // Load model
            guard let modelURL = Bundle.main.url(forResource: "EmotionClassificationModel", withExtension: "mlmodelc") else {
                print("Error: Model file not found in bundle")
                return nil
            }
            
            // Create configuration
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndGPU
            
            // Load model
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            let emotionClassifier = try EmotionClassificationModel(model: model)
            
            // Prepare image for model
            let grayscaleImage = convertToGrayscale(image)
            let modelSize = CGSize(width: 299, height: 299)
            guard let resizedImage = resizeImage(grayscaleImage, to: modelSize),
                  let pixelBuffer = pixelBuffer(from: resizedImage) else {
                return nil
            }
            
            // Make prediction
            return try emotionClassifier.prediction(image: pixelBuffer)
            
        } catch {
            print("Prediction error: \(error.localizedDescription)")
            return nil
        }
    }
    
    // MARK: - Helper Methods
    
    private func convertToGrayscale(_ image: UIImage) -> UIImage {
        let context = CIContext(options: nil)
        if let filter = CIFilter(name: "CIPhotoEffectMono") {
            filter.setValue(CIImage(image: image)!, forKey: kCIInputImageKey)
            if let output = filter.outputImage,
               let cgImage = context.createCGImage(output, from: output.extent) {
                return UIImage(cgImage: cgImage)
            }
        }
        return image
    }
    
    private func resizeImage(_ image: UIImage, to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 0.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resizedImage
    }
    
    private func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
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
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: pixelData,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        context?.translateBy(x: 0, y: size.height)
        context?.scaleBy(x: 1, y: -1)
        
        UIGraphicsPushContext(context!)
        image.draw(in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        UIGraphicsPopContext()
        
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        
        return pixelBuffer
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
                    self?.imageView.image = image
                    self?.selectedImage = image
                    self?.calculateScaledImageRect()
                    self?.processSelectedImage(image)
                }
            }
        }
    }
}

// MARK: - UIImagePicker Delegate
extension StillImageViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true) { [weak self] in
            if let image = info[.originalImage] as? UIImage {
                print("Image selected: \(image.size)")
                
                // Use main thread for UI updates
                DispatchQueue.main.async {
                    self?.imageView.image = image
                    self?.selectedImage = image
                    
                    // Calculate scaled rect AFTER setting the image
                    self?.calculateScaledImageRect()
                    
                    // Now process the image
                    self?.processSelectedImage(image)
                }
            }
        }
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
    }
}
