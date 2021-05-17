package com.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.location.Location;
import android.os.Build;
import android.os.Bundle;

import android.content.SharedPreferences;
import android.os.HandlerThread;
import android.util.Log;

import com.amplifyframework.AmplifyException;
import com.amplifyframework.core.Amplify;
import com.amplifyframework.core.model.query.Where;
import com.amplifyframework.datastore.AWSDataStorePlugin;
import com.amplifyframework.datastore.generated.model.User;
import com.cfm.recommend.Recommender;
import com.datastoreInteractions.AmplifyService;
import com.example.cfm.R;


import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.tasks.apmlify.SaveWorker;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkRequest;

import java.util.ArrayList;
import org.jetbrains.annotations.NotNull;
import com.spotifyFramework.Song;
import com.spotifyFramework.SongService;

public class MainActivity extends AppCompatActivity {

	private SharedPreferences.Editor editor;


	private Song song;
	private SongService songService;
	private ArrayList<Song> recentlyPlayed;
	private Recommender recommender;


    @RequiresApi(api = Build.VERSION_CODES.R)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        try {
            Amplify.addPlugin(new AWSDataStorePlugin());
            Amplify.configure(getApplicationContext());

            Log.i("MyAmplifyApp", "Initialized Amplify");
        } catch (AmplifyException error) {
            Log.e("MyAmplifyApp", "Could not initialize Amplify", error);
        }

        setContentView(R.layout.activity_main);

		songService = new SongService(getApplicationContext());
		recommender = new Recommender(getApplicationContext());
		SharedPreferences preferences = this.getSharedPreferences("SPOTIFY", 0);
		BottomNavigationView navView = findViewById(R.id.nav_view);
		// Passing each menu ID as a set of Ids because each
		// menu should be considered as top level destinations.
		AppBarConfiguration appBarConfiguration = new AppBarConfiguration.Builder(
				R.id.navigation_home, R.id.navigation_dashboard, R.id.song_dashboard,
				R.id.navigation_notifications)
				.build();
		NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment);
		NavigationUI.setupWithNavController(navView, navController);

		locationTest();
		getRecommendations();
		//amplifyTest();
	}

	private void getRecommendations() {
    	recommender.get("05", () -> {});
	}

/*
	private void amplifyTest() {
		WorkRequest saveWorkRequest = new OneTimeWorkRequest().Builder(User.class).build();

    	Amplify.DataStore.query(User.class, Where.id(user.getId()),
				matches -> {
    				if (matches.hasNext()) {
    					User theUser = matches.next();
    					Amplify.DataStore.delete(theUser,
								deleted -> Log.i("tests", "Deleted a user."),
								failure -> Log.e("tests", "Delete failed.", failure));
					}
				},
				failure -> Log.e("tests", "Query failed.", failure)
		);
	}
*/
	private void locationTest() {

		final long LOCATION_INTERVAL = 900000;
		final long FASTEST_LOCATION_INTERVAL = LOCATION_INTERVAL;

		LocationRequest lr = LocationRequest.create();
		lr.setInterval(LOCATION_INTERVAL);
		lr.setFastestInterval(FASTEST_LOCATION_INTERVAL);
		lr.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
		System.out.println(Build.VERSION.SDK_INT);

		final HandlerThread ht = new HandlerThread("Background location checker");
		ht.start();

		if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)
				!= PackageManager.PERMISSION_GRANTED || ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
				!= PackageManager.PERMISSION_GRANTED) {
			Log.e("Location perms", "Permissions were not granted.");
		}

		LocationServices.getFusedLocationProviderClient(this)
				.requestLocationUpdates(lr, new LocationCallback() {
					@Override
					public void onLocationResult(@NotNull LocationResult lr) {
						super.onLocationResult(lr);
						if (lr != null && lr.getLocations().size() > 0) {
							int i = lr.getLocations().size();
							Location loc = lr.getLocations().get(i - 1);
							//sendLocation(loc);
							//TODO implement only sending when necessary
						}
						//LocationServices.getFusedLocationProviderClient(MainActivity.this).removeLocationUpdates(this);
						//ht.quit();
					}
				}, ht.getLooper());
	}

	private void sendLocation(Location loc) {

    	AmplifyService as = new AmplifyService(getApplicationContext());
    	as.queryUser(getSharedPreferences("SPOTIFY", 0).getString("email", "no email address found"), null);	//TODO why a callback?
		as.updateLocation(loc.getLatitude(), loc.getLongitude());
		//TODO lemme find the schema
	}

}