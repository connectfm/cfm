package com.cfm.recommend;

import com.android.volley.Request;
import com.android.volley.toolbox.JsonObjectRequest;
import lombok.Builder;
import lombok.Value;

import com.spotifyFramework.VolleyCallBack;

/**
 * Provides song recommendations in the form of song URIs.
 */
@Value
@Builder
public class Recommender {

	public void get(final VolleyCallBack callBack) {
		String endpoint = "https://whateverourURLis.com/";
		JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(
				Request.Method.GET,
				endpoint,
				null,
				response -> {
					/*
					We'd get the call, and probably call the Spotify API for info
					On the URIs (I've already got a method set up for it in SongService)

					Then we store the now Song data structures in a List within this object

					After all that, we call the callback bc we know the data is compiled and
					properly stored
					 */

					/*
					RESPONSE TO ABOVE (rdt17):
						I think that is too much responsibility for a single class; too much coupling. Let
						this class just be responsible for giving you a song URI. Keeping the whole thing as
						a pipeline of transformations will help keep the logic organized. Using the Java
						functional API / lambda notation:
							1. Get recommended song URI: () -> recommender.recommend()
							2. Get song metadata (album image, artist, etc.): r -> getMetadata(r)
							3. Add song to the playback queue: s -> queue.add(s)
						I'm not sure if this actually how it is organized, but entangling the functionality
						like that will make debugging a nightmare. To go along with the SongService naming
						convention, the RecommendService, SongService, and PlaybackService should be
						completely oblivious to each other. They all do (or at least should do) different,
						independent tasks.
					 */
					callBack.onSuccess();

					/*
					Then we can do whatever we want down here to give info back to the server
					abt updated information (We could have prior info as a param)
					 */

					/*
					RESPONSE TO ABOVE (rdt17):
						There is nothing the server needs to know. All it does is gives recommendations. All
						data it receives is from ElastiCache, which is after its been saved in DataStore,
						synced to DynamoDB, and then processed by Alexs' Lambda function to be put into the
						cache.
					 */
				}, error -> {

				}
		);
	}

}
