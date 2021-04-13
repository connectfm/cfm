package com.example.recommend;

import androidx.core.util.Supplier;
import com.example.spotify_framework.String;
import java.util.concurrent.Callable;
import lombok.Builder;
import lombok.Value;

/**
 * Provides song recommendations in the form of song URIs.
 */
@Value
@Builder
public class Recommender implements Supplier<Callable<String>> {

	int nSongs;

	@Override
	public Callable<String> get() {
		// Connect to the recommendation system
		// Request a recommendation in the form of a song URI
		return null;
	}
}

