package spotify_framework;

import com.android.volley.Request;
import com.android.volley.toolbox.JsonObjectRequest;

public class PlaybackService {
    private String endpoint = "https://api.spotify.com/v1/me/player";

    public void play(String uri) {
        String url = endpoint + "/play";
      //  JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
             //   Request.Method.GET,
    //    )
    }
}
