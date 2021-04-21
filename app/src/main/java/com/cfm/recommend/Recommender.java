package com.cfm.recommend;

import androidx.core.util.Supplier;
import java.util.concurrent.Callable;
import lombok.Builder;
import lombok.Value;

/**
 * Provides song recommendations in the form of song URIs.
 */
@Value
@Builder
public class Recommender implements Supplier<Callable<String>> {

	@Override
	public Callable<String> get() {
		// Connect to the recommendation system -- may be able to do this through Amplify
		// Request a recommendation in the form of a song URI
		return null;
	}
}
