import UIKit
import CoreML

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Set up appearance
        setupAppearance()
        
        // Pre-warm the Vision and CoreML subsystems
        preWarmCoreML()
        
        return true
    }
    
    private func setupAppearance() {
        // Set up navigation bar appearance
        let appearance = UINavigationBar.appearance()
        appearance.tintColor = UIColor(named: "AccentColor")
        appearance.barTintColor = UIColor(named: "BackgroundColor")
        appearance.titleTextAttributes = [
            NSAttributedString.Key.foregroundColor: UIColor(named: "TextColor") ?? .black
        ]
        
        if #available(iOS 13.0, *) {
            let navBarAppearance = UINavigationBarAppearance()
            navBarAppearance.configureWithOpaqueBackground()
            navBarAppearance.backgroundColor = UIColor(named: "BackgroundColor")
            navBarAppearance.titleTextAttributes = [
                NSAttributedString.Key.foregroundColor: UIColor(named: "TextColor") ?? .black
            ]
            appearance.standardAppearance = navBarAppearance
            appearance.scrollEdgeAppearance = navBarAppearance
        }
        
        // Set up button appearance
        let buttonAppearance = UIButton.appearance()
        buttonAppearance.tintColor = UIColor(named: "AccentColor")
    }
    
    private func preWarmCoreML() {
        // Load model once to initialize the framework
        DispatchQueue.global(qos: .background).async {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndGPU
                
                if let modelURL = Bundle.main.url(forResource: "EmotionClassificationModel", withExtension: "mlmodelc") {
                    let _ = try MLModel(contentsOf: modelURL, configuration: config)
                    print("Model pre-warmed successfully")
                } else {
                    print("Model not found for pre-warming")
                }
            } catch {
                print("Model pre-warming failed: \(error)")
            }
        }
    }

    // MARK: UISceneSession Lifecycle

    @available(iOS 13.0, *)
    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    @available(iOS 13.0, *)
    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session
    }
}
