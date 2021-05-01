package ui;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import androidx.appcompat.app.AppCompatActivity;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.Volley;
import com.example.cfm.R;
import com.spotify.sdk.android.auth.AuthorizationClient;
import com.spotify.sdk.android.auth.AuthorizationRequest;
import com.spotify.sdk.android.auth.AuthorizationResponse;
import org.jetbrains.annotations.NotNull;
import pub.devrel.easypermissions.EasyPermissions;
import spotifyFramework.User;
import spotifyFramework.UserService;

public class LoginActivity extends AppCompatActivity {

	private static final String clientId = "9db9499ad1554b70b6942e9e3f3495e3";
	private static final String redirectUri = "https://example.com/callback/";
	private static final String[] scopes = new String[]{
			"user-read-email",
			"user-library-modify",
			"user-read-email",
			"user-read-private",
			"user-read-recently-played",
			"user-read-playback-state",
			"user-modify-playback-state"};
	private static final int reqCode = 0x10;
	private static final String TAG = "Spotify " + LoginActivity.class.getSimpleName();
	private SharedPreferences.Editor editor;
	private SharedPreferences preferences;
	private RequestQueue queue;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		setContentView(R.layout.activity_login);
		preferences = this.getSharedPreferences("SPOTIFY", 0);
		queue = Volley.newRequestQueue(this);

		checkPermissions();

		final Button button = findViewById(R.id.login_button);
		button.setOnClickListener(new View.OnClickListener() {
			public void onClick(View v) {
				authenticateSpotify(clientId, redirectUri, reqCode, scopes);
			}
		});
	}

	@Override
	protected void onActivityResult(int requestCode, final int resultCode, Intent data) {
		super.onActivityResult(requestCode, resultCode, data);

		if (requestCode == reqCode) {
			final AuthorizationResponse response = AuthorizationClient.getResponse(resultCode, data);

			switch (response.getType()) {
				// Response was successful and contains auth token
				case TOKEN:
					editor = getSharedPreferences("SPOTIFY", 0).edit();
					editor.putString("TOKEN", response.getAccessToken());
					editor.apply();
					waitForUserInfo();
					break;

				// Auth flow returned an error
				case ERROR:
					Log.e(TAG, "Auth error: " + response.getError());
					break;

				// Most likely auth flow was cancelled
				default:
					Log.d(TAG, "Auth result: " + response.getType());
			}
		}
	}

	private void waitForUserInfo() {
		UserService userService = new UserService(queue, preferences);
		userService.get(() -> {
			User user = userService.getUser();
			editor = getSharedPreferences("SPOTIFY", 0).edit();
			editor.putString("userid", user.id);
			editor.putString("displayName", user.display_name);
			editor.putString("country", user.country);
			Log.d("STARTING", "GOT USER INFORMATION");
			// We use commit instead of apply because we need the information stored immediately
			editor.apply();
			startMainActivity();
		});
	}

	private void startMainActivity() {
		Intent intent = new Intent(LoginActivity.this, MainActivity.class);
		startActivity(intent);
	}

	private void authenticateSpotify(String clientId, String redirectUri, int reqCode,
			String[] scopes) {
		AuthorizationRequest.Builder builder = new AuthorizationRequest.Builder(
				clientId,
				AuthorizationResponse.Type.TOKEN,
				redirectUri);

		builder.setScopes(scopes);
		AuthorizationRequest request = builder.build();
		AuthorizationClient.openLoginActivity(this, reqCode, request);
	}

	public void checkPermissions() {
		String[] perms = {
				Manifest.permission.ACCESS_FINE_LOCATION,
				Manifest.permission.ACCESS_COARSE_LOCATION,
				Manifest.permission.INTERNET,
				Manifest.permission.ACCESS_NETWORK_STATE,
				Manifest.permission.READ_EXTERNAL_STORAGE,
				Manifest.permission.WRITE_EXTERNAL_STORAGE
		};

		if (EasyPermissions.hasPermissions(this, perms)) {
			System.out.println("we have perms");
		} else {
			EasyPermissions.requestPermissions(this,
					"connect.fm requires location data. Please accept the following permission.", 1, perms);
		}
	}

	@Override
	public void onRequestPermissionsResult(int requestCode, @NotNull String[] permissions,
			@NotNull int[] grantResults) {
		super.onRequestPermissionsResult(requestCode, permissions, grantResults);

		// Forward results to EasyPermissions
		EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this);
	}
}