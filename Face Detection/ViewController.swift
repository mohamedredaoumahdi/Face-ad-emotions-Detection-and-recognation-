import UIKit

class ViewController: UIViewController {
    
    // MARK: - UI Components
    private let appNameLabel: UILabel = {
        let label = UILabel()
        label.text = "Face & Emotion Detection"
        label.font = UIFont.systemFont(ofSize: 24, weight: .bold)
        label.textAlignment = .center
        label.textColor = UIColor(named: "TextColor") ?? .black
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let appLogoImageView: UIImageView = {
        let imageView = UIImageView()
        imageView.image = UIImage(systemName: "face.smiling")
        imageView.contentMode = .scaleAspectFit
        imageView.tintColor = UIColor(named: "AccentColor") ?? .systemPurple
        imageView.translatesAutoresizingMaskIntoConstraints = false
        return imageView
    }()
    
    private let stillImageButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Analyze Image", for: .normal)
        button.setImage(UIImage(systemName: "photo"), for: .normal)
        button.tintColor = .white
        button.backgroundColor = UIColor(named: "AccentColor") ?? .systemPurple
        button.layer.cornerRadius = 16
        button.titleLabel?.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
        button.contentEdgeInsets = UIEdgeInsets(top: 16, left: 16, bottom: 16, right: 16)
        button.imageEdgeInsets = UIEdgeInsets(top: 0, left: 0, bottom: 0, right: 12)
        button.translatesAutoresizingMaskIntoConstraints = false
        
        // Add shadow
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 4)
        button.layer.shadowRadius = 8
        button.layer.shadowOpacity = 0.1
        return button
    }()
    
    private let liveFeedButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Live Emotion Detection", for: .normal)
        button.setImage(UIImage(systemName: "video"), for: .normal)
        button.tintColor = .white
        button.backgroundColor = UIColor(named: "AccentColor") ?? .systemPurple
        button.layer.cornerRadius = 16
        button.titleLabel?.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
        button.contentEdgeInsets = UIEdgeInsets(top: 16, left: 16, bottom: 16, right: 16)
        button.imageEdgeInsets = UIEdgeInsets(top: 0, left: 0, bottom: 0, right: 12)
        button.translatesAutoresizingMaskIntoConstraints = false
        
        // Add shadow
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 4)
        button.layer.shadowRadius = 8
        button.layer.shadowOpacity = 0.1
        return button
    }()
    
    private let aboutButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("About", for: .normal)
        button.setImage(UIImage(systemName: "info.circle"), for: .normal)
        button.tintColor = UIColor(named: "TextColor") ?? .darkGray
        button.backgroundColor = .clear
        button.layer.cornerRadius = 12
        button.titleLabel?.font = UIFont.systemFont(ofSize: 16, weight: .regular)
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    // MARK: - Lifecycle Methods
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
        setupActions()
        
        // Add subtle animation for the logo
        animateLogoOnAppear()
    }
    
    // MARK: - Setup Methods
    
    private func setupUI() {
        view.backgroundColor = UIColor(named: "BackgroundColor") ?? .systemBackground
        
        // Add subviews
        view.addSubview(appLogoImageView)
        view.addSubview(appNameLabel)
        view.addSubview(stillImageButton)
        view.addSubview(liveFeedButton)
        view.addSubview(aboutButton)
    }
    
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            // App logo
            appLogoImageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 40),
            appLogoImageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            appLogoImageView.widthAnchor.constraint(equalToConstant: 80),
            appLogoImageView.heightAnchor.constraint(equalToConstant: 80),
            
            // App name label
            appNameLabel.topAnchor.constraint(equalTo: appLogoImageView.bottomAnchor, constant: 16),
            appNameLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            appNameLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            
            // Live feed button
            liveFeedButton.topAnchor.constraint(equalTo: view.centerYAnchor, constant: -20),
            liveFeedButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            liveFeedButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            liveFeedButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
            liveFeedButton.heightAnchor.constraint(equalToConstant: 60),
            
            // Still image button
            stillImageButton.topAnchor.constraint(equalTo: liveFeedButton.bottomAnchor, constant: 20),
            stillImageButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            stillImageButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 40),
            stillImageButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -40),
            stillImageButton.heightAnchor.constraint(equalToConstant: 60),
            
            // About button
            aboutButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            aboutButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            aboutButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    private func setupActions() {
        stillImageButton.addTarget(self, action: #selector(didTapStillImage), for: .touchUpInside)
        liveFeedButton.addTarget(self, action: #selector(didTapLiveFeed), for: .touchUpInside)
        aboutButton.addTarget(self, action: #selector(didTapAbout), for: .touchUpInside)
    }
    
    // MARK: - Action Methods
    
    @objc private func didTapStillImage() {
        let stillImageVC = StillImageViewController()
        stillImageVC.modalPresentationStyle = .fullScreen
        navigationController?.pushViewController(stillImageVC, animated: true)
    }
    
    @objc private func didTapLiveFeed() {
        let liveFeedVC = LiveFeedViewController()
        liveFeedVC.modalPresentationStyle = .fullScreen
        navigationController?.pushViewController(liveFeedVC, animated: true)
    }
    
    @objc private func didTapAbout() {
        let alertController = UIAlertController(
            title: "About Face Detection",
            message: "This app was developed as a Master's graduation project in Computer Engineering for face detection and emotion recognition using Swift. The app uses Vision and CoreML to detect faces and predict emotions in real-time.",
            preferredStyle: .alert
        )
        alertController.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))
        present(alertController, animated: true, completion: nil)
    }
    
    // MARK: - Animation Methods
    
    private func animateLogoOnAppear() {
        // Start with 0 scale
        appLogoImageView.transform = CGAffineTransform(scaleX: 0.5, y: 0.5)
        appLogoImageView.alpha = 0
        
        // Animate to normal scale with spring effect
        UIView.animate(withDuration: 1.0, delay: 0.2, usingSpringWithDamping: 0.6, initialSpringVelocity: 0.5, options: [], animations: {
            self.appLogoImageView.transform = .identity
            self.appLogoImageView.alpha = 1
        }, completion: nil)
    }
}
