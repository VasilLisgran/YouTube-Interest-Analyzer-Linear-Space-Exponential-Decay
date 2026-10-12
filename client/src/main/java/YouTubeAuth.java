import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.extensions.java6.auth.oauth2.AuthorizationCodeInstalledApp;
import com.google.api.client.extensions.jetty.auth.oauth2.LocalServerReceiver;
import com.google.api.client.googleapis.auth.oauth2.GoogleAuthorizationCodeFlow;
import com.google.api.client.googleapis.auth.oauth2.GoogleClientSecrets;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.javanet.NetHttpTransport;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.services.youtube.YouTube;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.security.GeneralSecurityException;
import java.util.List;

public class YouTubeAuth {
    // Path with OAuth secrets
    private static final String CLIENT_SECRETS_PATH = "client-secret.json";
    // Factory for working with JSON (parsing API answers)
    private static final JsonFactory JSON_FACTORY = new JacksonFactory();

    // Scopes defines which data can we read (readonly)
    private static final List<String> SCOPES = List.of(
            "https://www.googleapis.com/auth/youtube.readonly",
            "https://www.googleapis.com/auth/youtube.force-ssl"  // History of liked videos
    );

    public static YouTube authenticate() throws IOException, GeneralSecurityException {
        // 1. NetHttpTransport is default Google HTTP-client
        NetHttpTransport httpTransport = GoogleNetHttpTransport.newTrustedTransport();

        InputStream in = YouTubeAuth.class.getResourceAsStream("/" + CLIENT_SECRETS_PATH);

        if (in == null) {
            in = YouTubeAuth.class.getClassLoader().
                    getResourceAsStream(CLIENT_SECRETS_PATH);
        }

        if (in == null) {
            throw new IOException("File not found in resources: " + CLIENT_SECRETS_PATH);
        }

        // 2. Loading secrets from JSON-file
        GoogleClientSecrets secrets = GoogleClientSecrets.load(
                JSON_FACTORY,
                new InputStreamReader(in)
        );

        // 3. Creating OAuth 2.0 flow (authorization process)
        // It works on codes and tokens
        GoogleAuthorizationCodeFlow flow = new GoogleAuthorizationCodeFlow.Builder(
                httpTransport,
                JSON_FACTORY,
                secrets,
                SCOPES
        )
                .setAccessType("offline")
                .build();

        // 4. Creating a local server on the port 8888
        // After login browser redirects to http://localhost:8888/callback
        // Takes a auth-code
        LocalServerReceiver receiver = new LocalServerReceiver.Builder()
                .setPort(8888) // CallBack port
                .build();

        // 5. Launching an authorizing process
        // - Opens browser with Google login page
        // - User logins and allows access
        // - Google redirects to the localhost with the code
        // - Local server catches the code
        // - Changing the code to access_token and refresh_token

        Credential credential = new AuthorizationCodeInstalledApp(flow, receiver)
                .authorize("user"); // ID user

        // 6. Create the YouTube service
        return new YouTube.Builder(httpTransport, JSON_FACTORY, credential)
                .setApplicationName("YouTube Recommender")  // Name of app
                .build();
    }

}

