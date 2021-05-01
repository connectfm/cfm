package spotifyFramework;

import android.content.SharedPreferences;
import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.JsonObjectRequest;
import com.google.gson.Gson;
import java.util.HashMap;
import java.util.Map;

public class UserService {

	private static final String ENDPOINT = "https://api.spotify.com/v1/me";
	private final SharedPreferences preferences;
	private final RequestQueue queue;
	private User user;

	public UserService(RequestQueue queue, SharedPreferences preferences) {
		this.queue = queue;
		this.preferences = preferences;
	}

	public User getUser() {
		return user;
	}

	public void get(final VolleyCallBack callBack) {
		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.GET,
				ENDPOINT,
				null,
				response -> {
					Gson gson = new Gson();
					user = gson.fromJson(response.toString(), User.class);
					callBack.onSuccess();
				}, error -> get(() -> {

		})) {
			@Override
			public Map<String, String> getHeaders() throws AuthFailureError {
				Map<String, String> headers = new HashMap<>();
				String token = preferences.getString("TOKEN", "");
				String auth = "Bearer " + token;
				headers.put("Authorization", auth);
				return headers;
			}
		};
		queue.add(jsonObjectRequest);
	}


}
