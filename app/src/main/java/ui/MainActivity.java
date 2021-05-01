package ui;

import android.Manifest;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.location.Location;
import android.os.Build;
import android.os.Bundle;
import android.os.HandlerThread;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;
import com.example.cfm.R;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import java.util.ArrayList;
import org.jetbrains.annotations.NotNull;
import spotifyFramework.Song;
import spotifyFramework.SongService;

public class MainActivity extends AppCompatActivity {

	private SharedPreferences.Editor editor;


	private Song song;
	private SongService songService;
	private ArrayList<Song> recentlyPlayed;


	@RequiresApi(api = Build.VERSION_CODES.R)
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);

		songService = new SongService(getApplicationContext());

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
	}

	private void locationTest() {

		final long LOCATION_INTERVAL = 900000;
		final long FASTEST_LOCATION_INTERVAL = LOCATION_INTERVAL;

		System.out.println("starting the test");
		LocationRequest lr = LocationRequest.create();
		lr.setInterval(LOCATION_INTERVAL);
		lr.setFastestInterval(FASTEST_LOCATION_INTERVAL);
		lr.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
		System.out.println(Build.VERSION.SDK_INT);

		final HandlerThread ht = new HandlerThread("location stuff");
		ht.start();

		if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION)
				!= PackageManager.PERMISSION_GRANTED) {
			System.out.println("damn pt 2");
		}

		LocationServices.getFusedLocationProviderClient(this)
				.requestLocationUpdates(lr, new LocationCallback() {
					@Override
					public void onLocationResult(@NotNull LocationResult lr) {
						super.onLocationResult(lr);
						System.out.println("in the locationcallback");
						if (lr != null && lr.getLocations().size() > 0) {
							int i = lr.getLocations().size();
							Location loc = lr.getLocations().get(i - 1);
							System.out.println(loc);
							sendLocation();
						}
						//LocationServices.getFusedLocationProviderClient(MainActivity.this).removeLocationUpdates(this);
						//ht.quit();
					}
				}, ht.getLooper());
		System.out.println("finishing the test");
	}

	private void sendLocation() {
		//TODO lemme find the schema
	}


}